import concurrent.futures
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    import lightning as L

    L.seed_everything(seed, workers=True)

    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_LABEL_NON_TARGET = "Non-Target"


def get_phase1_behavior(df: pl.DataFrame, sequence_id: str) -> str:
    """Get phase1 behavior for the specified sequence_id"""
    seq_behaviors = (
        df.filter(pl.col("sequence_id") == sequence_id).select("behavior").to_series().to_list()
    )

    phase1_behaviors = [
        "Moves hand to target location",
        "Relaxes and moves hand to target location",
    ]
    for behavior in seq_behaviors:
        if behavior in phase1_behaviors:
            return behavior

    raise ValueError(f"Phase1 behavior not found in sequence {sequence_id}")


def make_label_mapping(df: pl.DataFrame) -> dict[tuple[str, str, str], int]:
    """Create mapping from (orientation, gesture, phase1_behavior) to label index"""
    orientation_gesture_pairs = (
        df.filter(pl.col("sequence_counter") == 0)
        .select(["orientation", "gesture", "behavior"])
        .unique()
        .sort(["orientation", "gesture", "behavior"])
    )

    label2idx = {}
    for i, (orientation, gesture, phase1_behavior) in enumerate(
        orientation_gesture_pairs.iter_rows()
    ):
        label2idx[(orientation, gesture, phase1_behavior)] = i

    print(f"Created {len(label2idx)} orientation-gesture-phase1behavior combinations")

    orientations = df.select("orientation").unique().sort("orientation").to_series().to_list()
    for orientation in orientations:
        gestures_in_orientation = (
            df.filter(pl.col("orientation") == orientation)
            .select("gesture")
            .unique()
            .to_series()
            .to_list()
        )
        print(f"Orientation '{orientation}': {len(gestures_in_orientation)} gestures")

    return label2idx


def create_gesture_mapping_for_evaluation(
    df: pl.DataFrame, label2idx: dict[tuple[str, str, str], int]
) -> dict[str, Any]:
    """Create mapping to convert N-class predictions into 18-class evaluation space"""

    non_target_gestures = (
        df.filter(pl.col("sequence_type") == _LABEL_NON_TARGET)
        .select("gesture")
        .unique()
        .to_series()
        .to_list()
    )

    target_gestures = (
        df.filter(pl.col("sequence_type") == "Target")
        .select("gesture")
        .unique()
        .sort("gesture")
        .to_series()
        .to_list()
    )

    all_gestures = sorted(set(non_target_gestures + target_gestures))
    gesture_to_idx18 = {gesture: i for i, gesture in enumerate(all_gestures)}

    classN_to_class18 = {}
    for (_, gesture, _), idxN in label2idx.items():
        classN_to_class18[idxN] = gesture_to_idx18[gesture]

    class18_to_class9 = {}
    for gesture, idx18 in gesture_to_idx18.items():
        if gesture in non_target_gestures:
            class18_to_class9[idx18] = 0  # Non-Target
        else:
            target_idx = target_gestures.index(gesture) + 1
            class18_to_class9[idx18] = target_idx

    return {
        "class72_to_class18": classN_to_class18,
        "class18_to_class9": class18_to_class9,
        "non_target_gestures": non_target_gestures,
        "target_gestures": target_gestures,
        "non_target_indices": [gesture_to_idx18[g] for g in non_target_gestures],
        "gesture_to_idx18": gesture_to_idx18,
        "all_gestures": all_gestures,
    }


def prepare_dataframe(csv_path: str) -> pl.DataFrame:
    """Prepare and clean the dataframe"""
    csv_path_demo = csv_path.replace(".csv", "_demographics.csv")
    if not Path(csv_path).exists() or not Path(csv_path_demo).exists():
        input_dir = Path(csv_path).parent
        if not input_dir.exists():
            input_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["dvc", "pull"], check=True)

    df = pl.read_csv(csv_path)
    df_demo = pl.read_csv(csv_path_demo)
    df = df.join(df_demo, on="subject", how="left")

    keep_cols = [
        "row_id",
        "sequence_type",
        "sequence_id",
        "sequence_counter",
        "subject",
        "gesture",
        "behavior",
        "orientation",
        "acc_x",
        "acc_y",
        "acc_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "rot_w",
        "handedness",
    ]
    tof_cols = [f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)]
    keep_cols.extend(tof_cols)
    df = df.select(keep_cols)

    def fill_tof_col(col_name: str, df: pl.DataFrame) -> pl.Series:
        filled = (
            pl.when((pl.col(col_name).is_null()) | (pl.col(col_name) == -1))
            .then(0)
            .otherwise(pl.col(col_name))
            .alias(col_name)
        )
        return filled

    with concurrent.futures.ThreadPoolExecutor() as executor:
        filled_exprs = list(executor.map(lambda c: fill_tof_col(c, df), tof_cols))
    df = df.with_columns(filled_exprs)

    return df
