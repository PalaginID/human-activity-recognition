import subprocess
import sys
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import polars as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig
from torch import nn
from transformers.optimization import get_cosine_schedule_with_warmup

from .dataset import GestureDataModule, GestureDataset, MixupDataset
from .model import (
    ALL25DModel,
    ALLDeepModel,
    ALLModel,
    ALLSimpleModel,
    IMUDeepModel,
    IMUModel,
    IMUSimpleModel,
)
from .preproccessing import FEATURE_NAMES
from .utils import (
    create_gesture_mapping_for_evaluation,
    make_label_mapping,
    prepare_dataframe,
    set_seed,
)


class GestureLitModel(L.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        input_size = kwargs.get("input_size", 14)
        num_classes = kwargs.get("num_classes", 144)
        model_type = kwargs.get("model_type", "imu")
        eval_mapping = kwargs.get("eval_mapping", None)

        self.save_hyperparameters(ignore=["eval_mapping", "kwargs"])

        if model_type == "imu":
            self.model = IMUModel(
                input_size=input_size,
                n_classes=num_classes,
            )
        elif model_type == "all":
            self.model = ALLModel(
                input_size=input_size,
                n_classes=num_classes,
            )
        elif model_type == "imu_simple":
            self.model = IMUSimpleModel(
                input_size=input_size,
                n_classes=num_classes,
            )
        elif model_type == "all_simple":
            self.model = ALLSimpleModel(
                input_size=input_size,
                n_classes=num_classes,
            )
        elif model_type == "imu_deep":
            self.model = IMUDeepModel(
                input_size=input_size,
                n_classes=num_classes,
            )
        elif model_type == "all_deep":
            self.model = ALLDeepModel(
                input_size=input_size,
                n_classes=num_classes,
            )
        elif model_type == "all_25d":
            self.model = ALL25DModel(
                input_size=input_size,
                n_classes=num_classes,
            )
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        self.criterion = nn.CrossEntropyLoss()
        self.eval_mapping = eval_mapping

    def forward(self, x, lengths, phases=None):
        return self.model(x, lengths, phases)

    def _shared_step(self, batch, stage: str):
        x, lengths, y, phases = batch
        outputs = self(x, lengths, phases)

        if isinstance(outputs, dict):
            logits = outputs["gesture_logits"]
        else:
            logits = outputs

        if y.dim() > 1:  # soft label (mixup)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(y * log_probs).sum(dim=1).mean()
            y_hard = y.argmax(dim=1)
        else:  # hard label
            y_hard = y
            loss = self.criterion(logits, y)

        if isinstance(outputs, dict) and "phase_logits" in outputs:
            phase_criterion = nn.CrossEntropyLoss(ignore_index=-1)
            phase_loss = phase_criterion(outputs["phase_logits"].view(-1, 3), phases.view(-1))
            loss = loss + phase_loss
            self.log(f"{stage}/phase_loss", phase_loss, prog_bar=True, on_step=False, on_epoch=True)

        acc = (logits.argmax(dim=1) == y_hard).float().mean()

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        x, lengths, y, phases = batch
        outputs = self(x, lengths, phases)

        if isinstance(outputs, dict):
            logits = outputs["gesture_logits"]
        else:
            logits = outputs

        if y.dim() > 1:  # soft label
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(y * log_probs).sum(dim=1).mean()
            y_hard = y.argmax(dim=1)
        else:
            y_hard = y
            loss = self.criterion(logits, y)

        if isinstance(outputs, dict) and "phase_logits" in outputs:
            phase_criterion = nn.CrossEntropyLoss(ignore_index=-1)
            phase_loss = phase_criterion(outputs["phase_logits"].view(-1, 3), phases.view(-1))
            loss = loss + phase_loss
            self.log("val/phase_loss", phase_loss, prog_bar=True, on_step=False, on_epoch=True)

        acc = (logits.argmax(dim=1) == y_hard).float().mean()
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        preds = logits.argmax(dim=1)
        if not hasattr(self, "_val_preds"):
            self._val_preds = []
            self._val_targets = []
            self._val_logits = []
        self._val_preds.append(preds.cpu())
        self._val_targets.append(y_hard.cpu())
        self._val_logits.append(logits.cpu())
        return loss

    def on_validation_epoch_end(self):
        import numpy as _np
        from sklearn.metrics import f1_score as _f1

        if not hasattr(self, "_val_preds"):
            return

        preds_72 = torch.cat(self._val_preds).numpy()
        targets_72 = torch.cat(self._val_targets).numpy()
        val_logits = torch.cat(self._val_logits).numpy()

        if hasattr(self, "_logits_save_path") and self._logits_save_path:
            _np.save(
                self._logits_save_path,
                {"logits": val_logits, "targets": targets_72, "preds": preds_72},
            )

        if self.eval_mapping is not None:
            class72_to_class18 = self.eval_mapping["class72_to_class18"]
            class18_to_class9 = self.eval_mapping["class18_to_class9"]

            preds_18 = np.array([class72_to_class18[p] for p in preds_72])
            targets_18 = np.array([class72_to_class18[t] for t in targets_72])

            acc_18 = (preds_18 == targets_18).astype(float).mean()
            self.log("val/acc_18class", acc_18, prog_bar=True)

            preds_9 = np.array([class18_to_class9[p] for p in preds_18])
            targets_9 = np.array([class18_to_class9[t] for t in targets_18])

            f1_macro = _f1(targets_9, preds_9, average="macro")
            bin_targets = (targets_9 != 0).astype(int)
            bin_preds = (preds_9 != 0).astype(int)
            f1_binary = _f1(bin_targets, bin_preds)

            acc_72 = (preds_72 == targets_72).astype(float).mean()
            self.log("val/acc_72class", acc_72, prog_bar=False)
        else:
            f1_macro = _f1(targets_72, preds_72, average="macro")
            bin_targets = (targets_72 < 36).astype(int)
            bin_preds = (preds_72 < 36).astype(int)
            f1_binary = _f1(bin_targets, bin_preds)

        f1_mean = (f1_macro + f1_binary) / 2.0

        self.log("val/f1_macro", f1_macro, prog_bar=True)
        self.log("val/f1_binary", f1_binary, prog_bar=True)
        self.log("val/f1_mean", f1_mean, prog_bar=True)

        del self._val_preds
        del self._val_targets
        del self._val_logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * 0.1)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def prepare_ids(cfg: DictConfig, fold_idx: int, df: pl.DataFrame):
    seq_df = (
        df.select(["sequence_id", "subject", "orientation", "gesture", "handedness"])
        .unique()
        .sort(["sequence_id"])
    )
    seq_ids = seq_df["sequence_id"].to_list()
    subjects = seq_df["subject"].to_list()
    y = seq_df["handedness"].to_list()

    from sklearn.model_selection import StratifiedGroupKFold

    sgkf = StratifiedGroupKFold(
        n_splits=cfg.data.get("n_folds", 10), shuffle=True, random_state=cfg.get("seed", 42)
    )
    splits = list(sgkf.split(seq_ids, y=y, groups=subjects))
    train_idx, val_idx = splits[fold_idx]

    train_seq_ids = [seq_ids[i] for i in train_idx]
    val_seq_ids = [seq_ids[i] for i in val_idx]

    return train_seq_ids, val_seq_ids


def create_dataset(
    cfg: DictConfig,
    df: pl.DataFrame,
    train_seq_ids: list,
    val_seq_ids: list,
    label2idx: dict[tuple[str, str, str], int],
):
    use_tof = cfg.model.get("model_type", "imu") in ["all", "all_simple", "all_deep", "all_25d"]
    train_ds = GestureDataset(
        df,
        train_seq_ids,
        label2idx,
        use_tof=use_tof,
        use_tof_mask_augmentation_prob=cfg.train.get("use_tof_mask_augmentation_prob", 0.1),
        is_train=True,
        rot_zero=cfg.data.get("rot_zero", False),
    )
    val_ds = GestureDataset(
        df, val_seq_ids, label2idx, use_tof=use_tof, rot_zero=cfg.data.get("rot_zero", False)
    )

    if cfg.train.get("use_mixup", True):
        mixup_alpha = cfg.train.get("mixup_alpha", 0.5)
        print(f"Using Mixup with alpha={mixup_alpha}", flush=True)
        train_ds = MixupDataset(train_ds, alpha=mixup_alpha, num_classes=len(label2idx))

    return train_ds, val_ds


def log_git_info(run_id, tracking_uri):
    import mlflow

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], encoding="utf-8", stderr=subprocess.DEVNULL
    ).strip()
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("git.commit", commit)

    print(f"Training completed.\nMLflow run ID: {run_id}\nView results at: {tracking_uri}")


def run_fold(
    cfg: DictConfig, fold_idx: int, df: pl.DataFrame, label2idx: dict[tuple[str, str, str], int]
):
    print(f"Preparing data for fold {fold_idx} ...", flush=True)
    train_seq_ids, val_seq_ids = prepare_ids(cfg, fold_idx, df)

    print("Creating datasets ...", flush=True)
    train_ds, val_ds = create_dataset(cfg, df, train_seq_ids, val_seq_ids, label2idx)

    print("Initializing DataModule ...", flush=True)
    dm = GestureDataModule(
        train_ds,
        val_ds,
        batch_size=cfg.train.get("batch_size", 32),
        num_workers=cfg.data.get("num_workers", 32),
    )

    print("Building model ...", flush=True)
    eval_mapping = create_gesture_mapping_for_evaluation(df, label2idx)

    model = GestureLitModel(
        input_size=len(FEATURE_NAMES),
        hidden_size=cfg.model.get("hidden_size", 128),
        num_layers=cfg.model.get("num_layers", 2),
        num_classes=len(label2idx),
        lr=cfg.train.get("lr", 1e-3),
        weight_decay=cfg.train.get("weight_decay", 1e-4),
        model_type=cfg.model.get("model_type", "imu"),
        eval_mapping=eval_mapping,
    )

    base_out_dir = Path("../../output") / cfg.get("exp_name", "imu_102_10")
    run_idx = 0
    while (base_out_dir / str(run_idx)).exists():
        run_idx += 1
    out_dir = base_out_dir / str(run_idx)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    logits_save_path = out_dir / f"val_logits_fold{fold_idx}.npy"
    save_logits = cfg.train.get("save_logits", True)
    model._logits_save_path = str(logits_save_path) if save_logits else None

    if save_logits:
        val_seq_info_path = out_dir / f"val_seq_ids_fold{fold_idx}.npy"
        np.save(val_seq_info_path, val_seq_ids)

    debug = cfg.get("debug", False)
    if debug:
        mlflow_logger = False
        callbacks = []
    else:
        tracking_uri = cfg.logging.mlflow.get("tracking_uri", "http://127.0.0.1:8080")
        experiment_name = cfg.logging.mlflow.get("experiment_name", "gesture_recognition")

        mlflow_logger = MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            run_name=f"{cfg.get("exp_name", "imu_102_10")}_fold{fold_idx}",
        )

        checkpoint = ModelCheckpoint(
            dirpath=out_dir / "checkpoints",
            monitor="val/acc",
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_acc:.4f}",
            save_last=True,
        )

        callbacks = [
            LearningRateMonitor(logging_interval="epoch"),
            checkpoint,
        ]

    print("Setting up trainer ...", flush=True)
    loggers = [mlflow_logger] if mlflow_logger else []

    trainer = L.Trainer(
        accelerator=cfg.train.get("accelerator", "cpu"),
        devices=cfg.train.get("devices", 1),
        max_epochs=1 if debug else cfg.train.get("max_epochs", 50),
        logger=loggers,
        callbacks=callbacks,
        deterministic=True,
        fast_dev_run=1 if debug else False,
        default_root_dir=str(out_dir),
        enable_progress_bar=sys.stdout.isatty(),
        gradient_clip_val=1.0,
    )

    print("Starting training ...", flush=True)
    trainer.fit(model, dm)

    if mlflow_logger:
        log_git_info(mlflow_logger.run_id, tracking_uri)

    metrics = {}
    callback_metrics = trainer.callback_metrics
    for base in ["val/f1_macro", "val/f1_binary", "val/f1_mean", "val/acc"]:
        if base in callback_metrics:
            val = callback_metrics[base]
        elif f"{base}_epoch" in callback_metrics:
            val = callback_metrics[f"{base}_epoch"]
        else:
            continue

        if isinstance(val, torch.Tensor):
            val = val.item()
        metrics[base] = float(val)

    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function"""
    set_seed(cfg.get("seed", 42))

    csv_path = cfg.data.get("csv_path", "../../input/train.csv")
    print(f"Loading data from {csv_path} ...", flush=True)
    df = prepare_dataframe(csv_path)
    print("Dataframe prepared.", flush=True)

    print("Building label mapping ...", flush=True)
    label2idx = make_label_mapping(df)
    print(
        f"Label mapping prepared. Num orientation-gesture combinations: {len(label2idx)}",
        flush=True,
    )

    fold = cfg.data.get("fold", None)
    n_folds = cfg.data.get("n_folds", 10)
    folds_to_run = range(n_folds) if fold is None else [fold]

    fold_metrics_list = []
    for fold_idx in folds_to_run:
        print(f"\n===== Fold {fold_idx} / {n_folds} =====")
        metrics = run_fold(cfg, fold_idx, df, label2idx)
        fold_metrics_list.append(metrics)

    if fold_metrics_list:
        avg_metrics = {}
        all_keys = set().union(*(m.keys() for m in fold_metrics_list))
        for k in all_keys:
            vals = [m[k] for m in fold_metrics_list if k in m]
            if vals:
                avg_metrics[k] = float(np.mean(vals))

        print("\n===== Average validation metrics across folds =====")
        for k, v in avg_metrics.items():
            print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
