import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm


def create_mask(lengths, max_len):
    """Create mask from sequence lengths"""
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) < lengths.unsqueeze(1)
    return mask.float()  # (batch_size, max_len)


def masked_batch_norm(
    input: Tensor, mask: Tensor, weight: Tensor | None, bias: Tensor | None, **kwargs
) -> Tensor:
    # from https://gist.github.com/yangkky/364413426ec798589463a3a88be24219
    r"""Applies Masked Batch Normalization for each channel in each data sample in a batch.
    See :class:`~MaskedBatchNorm1d`, :class:`~MaskedBatchNorm2d`, :class:`~MaskedBatchNorm3d` for details.
    """
    running_mean = kwargs.get("running_mean")
    running_var = kwargs.get("running_var")
    training = kwargs.get("training")
    momentum = kwargs.get("momentum")
    eps = kwargs.get("eps", 1e-5)

    if not training and (running_mean is None or running_var is None):
        raise ValueError("Expected running_mean and running_var to be not None when training=False")

    num_dims = len(input.shape[2:])
    _dims = (0,) + tuple(range(-num_dims, 0))
    _slice = (None, ...) + (None,) * num_dims

    if training:
        num_elements = mask.sum(_dims)
        mean = (input * mask).sum(_dims) / num_elements  # (C,)
        var = (((input - mean[_slice]) * mask) ** 2).sum(_dims) / num_elements  # (C,)

        if running_mean is not None:
            running_mean.copy_(running_mean * (1 - momentum) + momentum * mean.detach())
        if running_var is not None:
            running_var.copy_(running_var * (1 - momentum) + momentum * var.detach())
    else:
        mean, var = running_mean, running_var

    out = (input - mean[_slice]) / torch.sqrt(var[_slice] + eps)  # (N, C, ...)

    if weight is not None and bias is not None:
        out = out * weight[_slice] + bias[_slice]

    return out


class _MaskedBatchNorm(_BatchNorm):
    # from https://gist.github.com/yangkky/364413426ec798589463a3a88be24219
    def __init__(self, num_features, **kwargs):
        super().__init__(
            num_features,
            kwargs.get("eps", 1e-5),
            kwargs.get("momentum", 0.1),
            kwargs.get("affine", True),
            kwargs.get("track_running_stats", True),
        )

    def forward(self, input: Tensor, mask: Tensor = None) -> Tensor:
        self._check_input_dim(input)
        if mask is not None:
            self._check_input_dim(mask)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        if mask is None:
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )
        else:
            return masked_batch_norm(
                input,
                mask,
                self.weight,
                self.bias,
                running_mean=self.running_mean
                if not self.training or self.track_running_stats
                else None,
                running_var=self.running_var
                if not self.training or self.track_running_stats
                else None,
                bn_training=bn_training,
                exponential_average_factor=exponential_average_factor,
                eps=self.eps,
            )


class MaskedBatchNorm1d(torch.nn.BatchNorm1d, _MaskedBatchNorm):
    # from https://gist.github.com/yangkky/364413426ec798589463a3a88be24219
    r"""Applies Batch Normalization over a masked 3D input
    (a mini-batch of 1D inputs with additional channel dimension)..
    See documentation of :class:`~torch.nn.BatchNorm1d` for details.
    Shape:
        - Input: :math:`(N, C, L)`
        - Mask: :math:`(N, 1, L)`
        - Output: :math:`(N, C, L)` (same shape as input)
    """

    def __init__(self, num_features: int, **kwargs) -> None:
        super().__init__(
            num_features,
            eps=kwargs.get("eps", 1e-5),
            momentum=kwargs.get("momentum", 0.1),
            affine=kwargs.get("affine", True),
            track_running_stats=kwargs.get("track_running_stats", True),
        )
        self.channels_last = kwargs.get("channels_last", False)

    def forward(self, inputs, mask=None):
        if self.channels_last:
            inputs = inputs.permute(0, 2, 1)
        if mask is not None:
            mask = mask[:, None, :]
        out = super().forward(inputs, mask)
        if self.channels_last:
            out = out.permute(0, 2, 1)
        return out


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, mask):
        # Compute masked average
        mask = mask.unsqueeze(1)  # (batch, 1, seq_len)
        masked_x = x * mask  # Zero out padding positions
        seq_lengths = mask.sum(dim=-1, keepdim=True)  # (batch, 1, 1)
        y = masked_x.sum(dim=-1, keepdim=True) / (seq_lengths + 1e-8)  # (batch, channels, 1)

        y = self.excitation(y.squeeze(-1)).unsqueeze(-1)  # (batch, channels, 1)
        return x * y.expand_as(x)


class ResidualSECNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        # First conv block
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.bn1 = MaskedBatchNorm1d(out_channels)

        # Second conv block
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.bn2 = MaskedBatchNorm1d(out_channels)

        # SE block
        self.se = SEBlock(out_channels)

        # Shortcut connection
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.shortcut_bn = MaskedBatchNorm1d(out_channels)

        self.dropout = nn.Dropout(kwargs.get("dropout", 0.3))

    def forward(self, x, mask):
        # Shortcut connection
        shortcut = self.shortcut(x) * mask.unsqueeze(1)
        shortcut = self.shortcut_bn(shortcut, mask)

        # First conv
        out = self.conv1(x) * mask.unsqueeze(1)
        out = F.relu(self.bn1(out, mask))
        # Second conv
        out = self.conv2(out) * mask.unsqueeze(1)
        out = self.bn2(out, mask)

        # SE block
        out = self.se(out, mask)

        # Add shortcut
        out += shortcut
        out = F.relu(out)

        out = self.dropout(out) * mask.unsqueeze(1)

        return out


class PhaseAttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, phase_weights, mask):
        # x shape: (batch, seq_len, hidden_dim)
        # phase_weights shape: (batch, seq_len)
        # mask shape: (batch, seq_len)

        scores = torch.tanh(self.attention(x))  # (batch, seq_len, 1)
        scores = scores.squeeze(-1)  # (batch, seq_len)

        # Apply mask by setting a large negative value for paddings
        scores = scores.masked_fill(~mask.bool(), -1e9)

        attention_weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # Combine with phase weights
        combined_weights = attention_weights * phase_weights  # (batch, seq_len)

        # Re-apply mask to zero out padding positions
        combined_weights = combined_weights * mask

        # Normalize to sum to 1
        combined_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + 1e-8)

        context = torch.sum(x * combined_weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return context


class IMUModel(nn.Module):
    def __init__(self, input_size, n_classes, weight_decay=1e-4):
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.weight_decay = weight_decay

        # IMU deep branch
        self.acc_block = nn.Sequential(
            ResidualSECNNBlock(3, 64, 1, dropout=0.3, weight_decay=weight_decay),
            ResidualSECNNBlock(64, 128, 3, dropout=0.3, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 256, 5, dropout=0.3, weight_decay=weight_decay),
        )
        self.rot_block = nn.Sequential(
            ResidualSECNNBlock(9, 64, 1, dropout=0.3, weight_decay=weight_decay),
            ResidualSECNNBlock(64, 128, 3, dropout=0.3, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 256, 5, dropout=0.3, weight_decay=weight_decay),
        )
        self.combined_block = nn.Sequential(
            ResidualSECNNBlock(3, 64, 1, dropout=0.3, weight_decay=weight_decay),
            ResidualSECNNBlock(64, 128, 3, dropout=0.3, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 256, 5, dropout=0.3, weight_decay=weight_decay),
        )
        self.imu_block = ResidualSECNNBlock(256 * 3, 256, 3, dropout=0.3, weight_decay=weight_decay)

        # Phase prediction head (3 phases)
        self.phase_head = nn.Linear(256, 3)  # phases: 0,1,2

        # Phase-aware attention pooling (for each of the 3 phases)
        self.phase1_attention = PhaseAttentionLayer(256)
        self.phase2_attention = PhaseAttentionLayer(256)
        self.phase3_attention = PhaseAttentionLayer(256)

        # Dense layers
        self.dense1 = nn.Linear(256 * 3, 512, bias=False)  # Concatenate features from 3 phases
        self.bn_dense1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(512, 256, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)

        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x, lengths, phases=None):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)

        # Create mask from lengths
        seq_len = x.size(2)
        mask = create_mask(lengths, seq_len)  # (batch, seq_len)

        # IMU branch (propagate mask to each block)
        x1 = self.acc_block[0](x[:, :3, :], mask)
        for layer in self.acc_block[1:]:
            x1 = layer(x1, mask)

        x2 = self.rot_block[0](x[:, 3:12, :], mask)
        for layer in self.rot_block[1:]:
            x2 = layer(x2, mask)

        x3 = self.combined_block[0](x[:, 12:, :], mask)
        for layer in self.combined_block[1:]:
            x3 = layer(x3, mask)

        x1 = self.imu_block(torch.cat([x1, x2, x3], dim=1), mask)

        x1 = x1.transpose(1, 2)  # (batch, seq_len, 256)

        # Phase prediction (apply mask)
        phase_logits = self.phase_head(x1)  # (batch, seq_len, 3)
        phase_logits = phase_logits.masked_fill(~mask.unsqueeze(-1).bool(), 0)
        phase_probs = F.softmax(phase_logits, dim=-1)  # (batch, seq_len, 3)

        # Phase-wise weights (ignore padding)
        phase1_weights = phase_probs[:, :, 0]  # (batch, seq_len)
        phase2_weights = phase_probs[:, :, 1]  # (batch, seq_len)
        phase3_weights = phase_probs[:, :, 2]  # (batch, seq_len)

        # Phase-wise attention pooling (pass mask)
        phase1_features = self.phase1_attention(x1, phase1_weights, mask)  # (batch, 256)
        phase2_features = self.phase2_attention(x1, phase2_weights, mask)  # (batch, 256)
        phase3_features = self.phase3_attention(x1, phase3_weights, mask)  # (batch, 256)

        # Concatenate features from three phases
        combined_features = torch.cat(
            [phase1_features, phase2_features, phase3_features], dim=-1
        )  # (batch, 768)

        # Dense layers
        x = F.relu(self.bn_dense1(self.dense1(combined_features)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)

        # Classification
        gesture_logits = self.classifier(x)

        # Return dictionary with both gesture and phase predictions
        return {"gesture_logits": gesture_logits, "phase_logits": phase_logits}


class ALLModel(nn.Module):
    def __init__(self, input_size, n_classes, weight_decay=1e-4):
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.weight_decay = weight_decay

        # IMU deep branch
        self.acc_block = nn.Sequential(
            ResidualSECNNBlock(3, 64, 1, dropout=0.3, weight_decay=weight_decay),
            ResidualSECNNBlock(64, 128, 3, dropout=0.3, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 256, 5, dropout=0.3, weight_decay=weight_decay),
        )
        self.rot_block = nn.Sequential(
            ResidualSECNNBlock(9, 64, 1, dropout=0.3, weight_decay=weight_decay),
            ResidualSECNNBlock(64, 128, 3, dropout=0.3, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 256, 5, dropout=0.3, weight_decay=weight_decay),
        )
        self.combined_block = nn.Sequential(
            ResidualSECNNBlock(3, 64, 1, dropout=0.3, weight_decay=weight_decay),
            ResidualSECNNBlock(64, 128, 3, dropout=0.3, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 256, 5, dropout=0.3, weight_decay=weight_decay),
        )
        self.tof_2d_block_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                )
                for _ in range(5)
            ]
        )
        self.tof_1d_block_list = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualSECNNBlock(64, 128, 3, dropout=0.3, weight_decay=weight_decay),
                    ResidualSECNNBlock(128, 256, 5, dropout=0.3, weight_decay=weight_decay),
                )
                for _ in range(5)
            ]
        )
        self.tof_block = ResidualSECNNBlock(256 * 5, 32, 3, dropout=0.3, weight_decay=weight_decay)
        self.imu_block = ResidualSECNNBlock(256 * 3, 256, 3, dropout=0.3, weight_decay=weight_decay)

        # Phase prediction head (3 phases)
        self.phase_head = nn.Linear(288, 3)  # phases: 0,1,2

        # Phase-aware attention pooling (for each of the 3 phases)
        self.phase1_attention = PhaseAttentionLayer(288)
        self.phase2_attention = PhaseAttentionLayer(288)
        self.phase3_attention = PhaseAttentionLayer(288)

        # Dense layers
        self.dense1 = nn.Linear(288 * 3, 512, bias=False)  # Concatenate features from 3 phases
        self.bn_dense1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(512, 256, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)

        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x, lengths, phases=None):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)

        # Create mask from lengths
        seq_len = x.size(2)
        mask = create_mask(lengths, seq_len)  # (batch, seq_len)

        # IMU branch (propagate mask to each block)
        x1 = self.acc_block[0](x[:, :3, :], mask)
        for layer in self.acc_block[1:]:
            x1 = layer(x1, mask)

        x2 = self.rot_block[0](x[:, 3:12, :], mask)
        for layer in self.rot_block[1:]:
            x2 = layer(x2, mask)

        x3 = self.combined_block[0](x[:, 12:15, :], mask)
        for layer in self.combined_block[1:]:
            x3 = layer(x3, mask)

        x1 = self.imu_block(torch.cat([x1, x2, x3], dim=1), mask)

        # ToF branch (respect mask)
        tof_features = []
        for i in range(5):
            x_tof = x[:, 15 + 64 * i : 15 + 64 * (i + 1), :]
            x_tof = x_tof.transpose(1, 2)  # (batch, seq_len, 64)
            B, S, _ = x_tof.size()

            # Use mask to zero out padding ToF data
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, 64)  # (B, S, 64)
            x_tof = x_tof * mask_expanded  # Zero out padding positions

            x_tof = x_tof.view(B * S, 1, 8, 8)
            x_tof = self.tof_2d_block_list[i](x_tof)
            x_tof = x_tof.mean(dim=(2, 3))  # (B*S, 64)
            x_tof = x_tof.view(B, S, -1)  # (B, S, 64)
            x_tof = x_tof * mask_expanded
            x_tof = x_tof.transpose(1, 2)  # (B, 64, S)

            # Pass mask to 1D block
            for layer in self.tof_1d_block_list[i]:
                x_tof = layer(x_tof, mask)
            tof_features.append(x_tof)

        x_tof = self.tof_block(torch.cat(tof_features, dim=1), mask)
        x1 = torch.cat([x1, x_tof], dim=1)
        x1 = x1.transpose(1, 2)  # (batch, seq_len, 512)

        # Phase prediction (apply mask)
        phase_logits = self.phase_head(x1)  # (batch, seq_len, 3)
        phase_logits = phase_logits.masked_fill(~mask.unsqueeze(-1).bool(), 0)
        phase_probs = F.softmax(phase_logits, dim=-1)  # (batch, seq_len, 3)

        # Phase-wise weights (ignore padding)
        phase1_weights = phase_probs[:, :, 0]  # (batch, seq_len)
        phase2_weights = phase_probs[:, :, 1]  # (batch, seq_len)
        phase3_weights = phase_probs[:, :, 2]  # (batch, seq_len)

        # Phase-wise attention pooling (pass mask)
        phase1_features = self.phase1_attention(x1, phase1_weights, mask)  # (batch, 512)
        phase2_features = self.phase2_attention(x1, phase2_weights, mask)  # (batch, 512)
        phase3_features = self.phase3_attention(x1, phase3_weights, mask)  # (batch, 512)

        # Concatenate features from three phases
        combined_features = torch.cat(
            [phase1_features, phase2_features, phase3_features], dim=-1
        )  # (batch, 1536)

        # Dense layers
        x = F.relu(self.bn_dense1(self.dense1(combined_features)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)

        # Classification
        gesture_logits = self.classifier(x)

        # Return dictionary with both gesture and phase predictions
        return {"gesture_logits": gesture_logits, "phase_logits": phase_logits}


class IMUSimpleModel(nn.Module):
    def __init__(self, input_size, n_classes, weight_decay=1e-4):
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.weight_decay = weight_decay

        # IMU deep branch
        self.acc_block = nn.Sequential(
            ResidualSECNNBlock(3, 256, 3, dropout=0.3, weight_decay=weight_decay),
        )
        self.rot_block = nn.Sequential(
            ResidualSECNNBlock(9, 256, 3, dropout=0.3, weight_decay=weight_decay),
        )
        self.combined_block = nn.Sequential(
            ResidualSECNNBlock(3, 256, 3, dropout=0.3, weight_decay=weight_decay),
        )
        self.imu_block = ResidualSECNNBlock(256 * 3, 256, 5, dropout=0.3, weight_decay=weight_decay)

        # Phase prediction head (3 phases)
        self.phase_head = nn.Linear(256, 3)  # phases: 0,1,2

        # Phase-aware attention pooling (for each of the 3 phases)
        self.phase1_attention = PhaseAttentionLayer(256)
        self.phase2_attention = PhaseAttentionLayer(256)
        self.phase3_attention = PhaseAttentionLayer(256)

        # Dense layers
        self.dense1 = nn.Linear(256 * 3, 512, bias=False)  # Concatenate features from 3 phases
        self.bn_dense1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(512, 256, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)

        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x, lengths, phases=None):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)

        # Create mask from lengths
        seq_len = x.size(2)
        mask = create_mask(lengths, seq_len)  # (batch, seq_len)

        # IMU branch (propagate mask to each block)
        x1 = self.acc_block[0](x[:, :3, :], mask)

        x2 = self.rot_block[0](x[:, 3:12, :], mask)

        x3 = self.combined_block[0](x[:, 12:, :], mask)

        x1 = self.imu_block(torch.cat([x1, x2, x3], dim=1), mask)

        x1 = x1.transpose(1, 2)  # (batch, seq_len, 256)

        # Phase prediction (apply mask)
        phase_logits = self.phase_head(x1)  # (batch, seq_len, 3)
        phase_logits = phase_logits.masked_fill(~mask.unsqueeze(-1).bool(), 0)
        phase_probs = F.softmax(phase_logits, dim=-1)  # (batch, seq_len, 3)

        # Phase-wise weights (ignore padding)
        phase1_weights = phase_probs[:, :, 0]  # (batch, seq_len)
        phase2_weights = phase_probs[:, :, 1]  # (batch, seq_len)
        phase3_weights = phase_probs[:, :, 2]  # (batch, seq_len)

        # Phase-wise attention pooling (pass mask)
        phase1_features = self.phase1_attention(x1, phase1_weights, mask)  # (batch, 256)
        phase2_features = self.phase2_attention(x1, phase2_weights, mask)  # (batch, 256)
        phase3_features = self.phase3_attention(x1, phase3_weights, mask)  # (batch, 256)

        # Concatenate features from three phases
        combined_features = torch.cat(
            [phase1_features, phase2_features, phase3_features], dim=-1
        )  # (batch, 768)

        # Dense layers
        x = F.relu(self.bn_dense1(self.dense1(combined_features)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)

        # Classification
        gesture_logits = self.classifier(x)

        # Return dictionary with both gesture and phase predictions
        return {"gesture_logits": gesture_logits, "phase_logits": phase_logits}


class ALLSimpleModel(nn.Module):
    def __init__(self, input_size, n_classes, weight_decay=1e-4):
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.weight_decay = weight_decay

        # IMU deep branch
        self.acc_block = nn.Sequential(
            ResidualSECNNBlock(3, 256, 3, dropout=0.3, weight_decay=weight_decay),
        )
        self.rot_block = nn.Sequential(
            ResidualSECNNBlock(9, 256, 3, dropout=0.3, weight_decay=weight_decay),
        )
        self.combined_block = nn.Sequential(
            ResidualSECNNBlock(3, 256, 3, dropout=0.3, weight_decay=weight_decay),
        )
        self.tof_2d_block_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                )
                for _ in range(5)
            ]
        )
        self.tof_1d_block_list = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualSECNNBlock(64, 256, 3, dropout=0.3, weight_decay=weight_decay),
                )
                for _ in range(5)
            ]
        )
        self.tof_block = ResidualSECNNBlock(256 * 5, 32, 5, dropout=0.3, weight_decay=weight_decay)
        self.imu_block = ResidualSECNNBlock(256 * 3, 256, 5, dropout=0.3, weight_decay=weight_decay)

        # Phase prediction head (3 phases)
        self.phase_head = nn.Linear(288, 3)  # phases: 0,1,2

        # Phase-aware attention pooling (for each of the 3 phases)
        self.phase1_attention = PhaseAttentionLayer(288)
        self.phase2_attention = PhaseAttentionLayer(288)
        self.phase3_attention = PhaseAttentionLayer(288)

        # Dense layers
        self.dense1 = nn.Linear(288 * 3, 512, bias=False)  # Concatenate features from 3 phases
        self.bn_dense1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(512, 256, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)

        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x, lengths, phases=None):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)

        # Create mask from lengths
        seq_len = x.size(2)
        mask = create_mask(lengths, seq_len)  # (batch, seq_len)

        # IMU branch (propagate mask to each block)
        x1 = self.acc_block[0](x[:, :3, :], mask)

        x2 = self.rot_block[0](x[:, 3:12, :], mask)

        x3 = self.combined_block[0](x[:, 12:15, :], mask)

        x1 = self.imu_block(torch.cat([x1, x2, x3], dim=1), mask)

        # ToF branch (respect mask)
        tof_features = []
        for i in range(5):
            x_tof = x[:, 15 + 64 * i : 15 + 64 * (i + 1), :]
            x_tof = x_tof.transpose(1, 2)  # (batch, seq_len, 64)
            B, S, _ = x_tof.size()

            # Use mask to zero out padding ToF data
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, 64)  # (B, S, 64)
            x_tof = x_tof * mask_expanded  # Zero out padding positions

            x_tof = x_tof.view(B * S, 1, 8, 8)
            x_tof = self.tof_2d_block_list[i](x_tof)
            x_tof = x_tof.mean(dim=(2, 3))  # (B*S, 64)
            x_tof = x_tof.view(B, S, -1)  # (B, S, 64)
            x_tof = x_tof * mask_expanded
            x_tof = x_tof.transpose(1, 2)  # (B, 64, S)

            # Pass mask to 1D block
            for layer in self.tof_1d_block_list[i]:
                x_tof = layer(x_tof, mask)
            tof_features.append(x_tof)

        x_tof = self.tof_block(torch.cat(tof_features, dim=1), mask)
        x1 = torch.cat([x1, x_tof], dim=1)
        x1 = x1.transpose(1, 2)  # (batch, seq_len, 512)

        # Phase prediction (apply mask)
        phase_logits = self.phase_head(x1)  # (batch, seq_len, 3)
        phase_logits = phase_logits.masked_fill(~mask.unsqueeze(-1).bool(), 0)
        phase_probs = F.softmax(phase_logits, dim=-1)  # (batch, seq_len, 3)

        # Phase-wise weights (ignore padding)
        phase1_weights = phase_probs[:, :, 0]  # (batch, seq_len)
        phase2_weights = phase_probs[:, :, 1]  # (batch, seq_len)
        phase3_weights = phase_probs[:, :, 2]  # (batch, seq_len)

        # Phase-wise attention pooling (pass mask)
        phase1_features = self.phase1_attention(x1, phase1_weights, mask)  # (batch, 512)
        phase2_features = self.phase2_attention(x1, phase2_weights, mask)  # (batch, 512)
        phase3_features = self.phase3_attention(x1, phase3_weights, mask)  # (batch, 512)

        # Concatenate features from three phases
        combined_features = torch.cat(
            [phase1_features, phase2_features, phase3_features], dim=-1
        )  # (batch, 1536)

        # Dense layers
        x = F.relu(self.bn_dense1(self.dense1(combined_features)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)

        # Classification
        gesture_logits = self.classifier(x)

        # Return dictionary with both gesture and phase predictions
        return {"gesture_logits": gesture_logits, "phase_logits": phase_logits}


class IMUDeepModel(nn.Module):
    def __init__(self, input_size, n_classes, weight_decay=1e-4):
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.weight_decay = weight_decay

        # IMU deep branch
        self.acc_block = nn.Sequential(
            ResidualSECNNBlock(3, 128, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
        )
        self.rot_block = nn.Sequential(
            ResidualSECNNBlock(9, 128, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
        )
        self.combined_block = nn.Sequential(
            ResidualSECNNBlock(3, 128, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
        )
        self.imu_block = nn.Sequential(
            ResidualSECNNBlock(128 * 3, 256, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(256, 256, 3, dropout=0.1, weight_decay=weight_decay),
        )

        # Phase prediction head (3 phases)
        self.phase_head = nn.Linear(256, 3)  # phases: 0,1,2

        # Phase-aware attention pooling (for each of the 3 phases)
        self.phase1_attention = PhaseAttentionLayer(256)
        self.phase2_attention = PhaseAttentionLayer(256)
        self.phase3_attention = PhaseAttentionLayer(256)

        # Dense layers
        self.dense1 = nn.Linear(256 * 3, 512, bias=False)  # Concatenate features from 3 phases
        self.bn_dense1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(512, 256, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)

        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x, lengths, phases=None):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)

        # Create mask from lengths
        seq_len = x.size(2)
        mask = create_mask(lengths, seq_len)  # (batch, seq_len)

        # IMU branch (propagate mask to each block)
        x1 = self.acc_block[0](x[:, :3, :], mask)
        for layer in self.acc_block[1:]:
            x1 = layer(x1, mask)

        x2 = self.rot_block[0](x[:, 3:12, :], mask)
        for layer in self.rot_block[1:]:
            x2 = layer(x2, mask)

        x3 = self.combined_block[0](x[:, 12:, :], mask)
        for layer in self.combined_block[1:]:
            x3 = layer(x3, mask)

        x1 = self.imu_block[0](torch.cat([x1, x2, x3], dim=1), mask)
        x1 = self.imu_block[1](x1, mask)

        x1 = x1.transpose(1, 2)  # (batch, seq_len, 256)

        # Phase prediction (apply mask)
        phase_logits = self.phase_head(x1)  # (batch, seq_len, 3)
        phase_logits = phase_logits.masked_fill(~mask.unsqueeze(-1).bool(), 0)
        phase_probs = F.softmax(phase_logits, dim=-1)  # (batch, seq_len, 3)

        # Phase-wise weights (ignore padding)
        phase1_weights = phase_probs[:, :, 0]  # (batch, seq_len)
        phase2_weights = phase_probs[:, :, 1]  # (batch, seq_len)
        phase3_weights = phase_probs[:, :, 2]  # (batch, seq_len)

        # Phase-wise attention pooling (pass mask)
        phase1_features = self.phase1_attention(x1, phase1_weights, mask)  # (batch, 256)
        phase2_features = self.phase2_attention(x1, phase2_weights, mask)  # (batch, 256)
        phase3_features = self.phase3_attention(x1, phase3_weights, mask)  # (batch, 256)

        # Concatenate features from three phases
        combined_features = torch.cat(
            [phase1_features, phase2_features, phase3_features], dim=-1
        )  # (batch, 768)

        # Dense layers
        x = F.relu(self.bn_dense1(self.dense1(combined_features)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)

        # Classification
        gesture_logits = self.classifier(x)

        # Return dictionary with both gesture and phase predictions
        return {"gesture_logits": gesture_logits, "phase_logits": phase_logits}


class ALLDeepModel(nn.Module):
    def __init__(self, input_size, n_classes, weight_decay=1e-4):
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.weight_decay = weight_decay

        # IMU deep branch
        self.acc_block = nn.Sequential(
            ResidualSECNNBlock(3, 128, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
        )
        self.rot_block = nn.Sequential(
            ResidualSECNNBlock(9, 128, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
        )
        self.combined_block = nn.Sequential(
            ResidualSECNNBlock(3, 128, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
        )
        self.tof_2d_block_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                )
                for _ in range(5)
            ]
        )
        self.tof_1d_block_list = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualSECNNBlock(64, 128, 3, dropout=0.1, weight_decay=weight_decay),
                    ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
                    ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
                    ResidualSECNNBlock(128, 128, 3, dropout=0.1, weight_decay=weight_decay),
                )
                for _ in range(5)
            ]
        )
        self.tof_block = nn.Sequential(
            ResidualSECNNBlock(128 * 5, 128, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 32, 3, dropout=0.1, weight_decay=weight_decay),
        )
        self.imu_block = nn.Sequential(
            ResidualSECNNBlock(128 * 3, 256, 3, dropout=0.1, weight_decay=weight_decay),
            ResidualSECNNBlock(256, 256, 3, dropout=0.1, weight_decay=weight_decay),
        )

        # Phase prediction head (3 phases)
        self.phase_head = nn.Linear(288, 3)  # phases: 0,1,2

        # Phase-aware attention pooling (for each of the 3 phases)
        self.phase1_attention = PhaseAttentionLayer(288)
        self.phase2_attention = PhaseAttentionLayer(288)
        self.phase3_attention = PhaseAttentionLayer(288)

        # Dense layers
        self.dense1 = nn.Linear(288 * 3, 512, bias=False)  # Concatenate features from 3 phases
        self.bn_dense1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(512, 256, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)

        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x, lengths, phases=None):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)

        # Create mask from lengths
        seq_len = x.size(2)
        mask = create_mask(lengths, seq_len)  # (batch, seq_len)

        # IMU branch (propagate mask to each block)
        x1 = self.acc_block[0](x[:, :3, :], mask)
        for layer in self.acc_block[1:]:
            x1 = layer(x1, mask)

        x2 = self.rot_block[0](x[:, 3:12, :], mask)
        for layer in self.rot_block[1:]:
            x2 = layer(x2, mask)

        x3 = self.combined_block[0](x[:, 12:15, :], mask)
        for layer in self.combined_block[1:]:
            x3 = layer(x3, mask)

        x1 = self.imu_block[0](torch.cat([x1, x2, x3], dim=1), mask)
        x1 = self.imu_block[1](x1, mask)

        # ToF branch (respect mask)
        tof_features = self._create_tof_features(x, mask)

        x_tof = self.tof_block[0](torch.cat(tof_features, dim=1), mask)
        x_tof = self.tof_block[1](x_tof, mask)
        x1 = torch.cat([x1, x_tof], dim=1)
        x1 = x1.transpose(1, 2)  # (batch, seq_len, 512)

        # Phase prediction (apply mask)
        phase_logits = self.phase_head(x1)  # (batch, seq_len, 3)
        phase_logits = phase_logits.masked_fill(~mask.unsqueeze(-1).bool(), 0)
        phase_probs = F.softmax(phase_logits, dim=-1)  # (batch, seq_len, 3)

        # Phase-wise weights (ignore padding)
        phase1_weights = phase_probs[:, :, 0]  # (batch, seq_len)
        phase2_weights = phase_probs[:, :, 1]  # (batch, seq_len)
        phase3_weights = phase_probs[:, :, 2]  # (batch, seq_len)

        # Phase-wise attention pooling (pass mask)
        phase1_features = self.phase1_attention(x1, phase1_weights, mask)  # (batch, 512)
        phase2_features = self.phase2_attention(x1, phase2_weights, mask)  # (batch, 512)
        phase3_features = self.phase3_attention(x1, phase3_weights, mask)  # (batch, 512)

        # Concatenate features from three phases
        combined_features = torch.cat(
            [phase1_features, phase2_features, phase3_features], dim=-1
        )  # (batch, 1536)

        # Dense layers
        x = F.relu(self.bn_dense1(self.dense1(combined_features)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)

        # Classification
        gesture_logits = self.classifier(x)

        # Return dictionary with both gesture and phase predictions
        return {"gesture_logits": gesture_logits, "phase_logits": phase_logits}

    def _create_tof_features(self, x, mask):
        tof_features = []
        for i in range(5):
            x_tof = x[:, 15 + 64 * i : 15 + 64 * (i + 1), :]
            x_tof = x_tof.transpose(1, 2)  # (batch, seq_len, 64)
            B, S, _ = x_tof.size()

            # Use mask to zero out padding ToF data
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, 64)  # (B, S, 64)
            x_tof = x_tof * mask_expanded  # Zero out padding positions

            x_tof = x_tof.view(B * S, 1, 8, 8)
            x_tof = self.tof_2d_block_list[i](x_tof)
            x_tof = x_tof.mean(dim=(2, 3))  # (B*S, 64)
            x_tof = x_tof.view(B, S, -1)  # (B, S, 64)
            x_tof = x_tof * mask_expanded
            x_tof = x_tof.transpose(1, 2)  # (B, 64, S)

            # Pass mask to 1D block
            for layer in self.tof_1d_block_list[i]:
                x_tof = layer(x_tof, mask)
            tof_features.append(x_tof)

        return tof_features


class ALL25DModel(nn.Module):
    def __init__(self, input_size, n_classes, weight_decay=1e-4):
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.weight_decay = weight_decay

        # IMU deep branch
        self.acc_block = nn.Sequential(
            ResidualSECNNBlock(3, 64, 1, dropout=0.3, weight_decay=weight_decay),
            ResidualSECNNBlock(64, 128, 3, dropout=0.3, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 256, 5, dropout=0.3, weight_decay=weight_decay),
        )
        self.rot_block = nn.Sequential(
            ResidualSECNNBlock(9, 64, 1, dropout=0.3, weight_decay=weight_decay),
            ResidualSECNNBlock(64, 128, 3, dropout=0.3, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 256, 5, dropout=0.3, weight_decay=weight_decay),
        )
        self.combined_block = nn.Sequential(
            ResidualSECNNBlock(3, 64, 1, dropout=0.3, weight_decay=weight_decay),
            ResidualSECNNBlock(64, 128, 3, dropout=0.3, weight_decay=weight_decay),
            ResidualSECNNBlock(128, 256, 5, dropout=0.3, weight_decay=weight_decay),
        )
        self.tof_2d_block_list1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.AvgPool2d(2),
                    nn.Dropout(0.3),
                )
                for _ in range(5)
            ]
        )
        self.tof_2d_block_list2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.AvgPool2d(2),
                    nn.Dropout(0.3),
                )
                for _ in range(5)
            ]
        )
        self.tof_1d_block_list1 = nn.ModuleList(
            [ResidualSECNNBlock(1, 16, 1, dropout=0.3, weight_decay=weight_decay) for _ in range(5)]
        )
        self.tof_1d_block_list2 = nn.ModuleList(
            [
                ResidualSECNNBlock(32, 64, 3, dropout=0.3, weight_decay=weight_decay)
                for _ in range(5)
            ]
        )
        self.tof_1d_block_list3 = nn.ModuleList(
            [
                ResidualSECNNBlock(128, 256, 5, dropout=0.3, weight_decay=weight_decay)
                for _ in range(5)
            ]
        )
        self.tof_block = ResidualSECNNBlock(256 * 5, 32, 3, dropout=0.3, weight_decay=weight_decay)
        self.imu_block = ResidualSECNNBlock(256 * 3, 256, 3, dropout=0.3, weight_decay=weight_decay)

        # Phase prediction head (3 phases)
        self.phase_head = nn.Linear(288, 3)  # phases: 0,1,2

        # Phase-aware attention pooling (for each of the 3 phases)
        self.phase1_attention = PhaseAttentionLayer(288)
        self.phase2_attention = PhaseAttentionLayer(288)
        self.phase3_attention = PhaseAttentionLayer(288)

        # Dense layers
        self.dense1 = nn.Linear(288 * 3, 512, bias=False)  # Concatenate features from 3 phases
        self.bn_dense1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(512, 256, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)

        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x, lengths, phases=None):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)

        # Create mask from lengths
        seq_len = x.size(2)
        mask = create_mask(lengths, seq_len)  # (batch, seq_len)

        # IMU branch (propagate mask to each block)
        x1 = self.acc_block[0](x[:, :3, :], mask)
        for layer in self.acc_block[1:]:
            x1 = layer(x1, mask)

        x2 = self.rot_block[0](x[:, 3:12, :], mask)
        for layer in self.rot_block[1:]:
            x2 = layer(x2, mask)

        x3 = self.combined_block[0](x[:, 12:15, :], mask)
        for layer in self.combined_block[1:]:
            x3 = layer(x3, mask)

        x1 = self.imu_block(torch.cat([x1, x2, x3], dim=1), mask)

        # ToF branch (respect mask)
        tof_features = self._create_tof_features(self, x, mask)

        x_tof = self.tof_block(torch.cat(tof_features, dim=1), mask)
        x1 = torch.cat([x1, x_tof], dim=1)
        x1 = x1.transpose(1, 2)  # (batch, seq_len, 512)

        # Phase prediction (apply mask)
        phase_logits = self.phase_head(x1)  # (batch, seq_len, 3)
        phase_logits = phase_logits.masked_fill(~mask.unsqueeze(-1).bool(), 0)
        phase_probs = F.softmax(phase_logits, dim=-1)  # (batch, seq_len, 3)

        # Phase-wise weights (ignore padding)
        phase1_weights = phase_probs[:, :, 0]  # (batch, seq_len)
        phase2_weights = phase_probs[:, :, 1]  # (batch, seq_len)
        phase3_weights = phase_probs[:, :, 2]  # (batch, seq_len)

        # Phase-wise attention pooling (pass mask)
        phase1_features = self.phase1_attention(x1, phase1_weights, mask)  # (batch, 512)
        phase2_features = self.phase2_attention(x1, phase2_weights, mask)  # (batch, 512)
        phase3_features = self.phase3_attention(x1, phase3_weights, mask)  # (batch, 512)

        # Concatenate features from three phases
        combined_features = torch.cat(
            [phase1_features, phase2_features, phase3_features], dim=-1
        )  # (batch, 1536)

        # Dense layers
        x = F.relu(self.bn_dense1(self.dense1(combined_features)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)

        # Classification
        gesture_logits = self.classifier(x)

        # Return dictionary with both gesture and phase predictions
        return {"gesture_logits": gesture_logits, "phase_logits": phase_logits}

    def _create_tof_features(self, x, mask):
        tof_features = []
        for i in range(5):
            x_tof = x[:, 15 + 64 * i : 15 + 64 * (i + 1), :]
            B, C, S = x_tof.size()
            x_tof = x_tof.reshape(B * 64, 1, S)  # (B*64, 1, S)
            mask_expanded = mask.unsqueeze(1).repeat(1, 64, 1).reshape(B * 64, S)  # (B*64, S)
            x_tof = self.tof_1d_block_list1[i](x_tof, mask_expanded)  # (B*64, 16, S)
            x_tof = x_tof.reshape(B, 64, 16, S)
            x_tof = x_tof.transpose(1, 3)  # (B, S, 16, 64)
            x_tof = x_tof.reshape(B * S, 16, 8, 8)  # (B*S, 16, 8, 8)
            x_tof = self.tof_2d_block_list1[i](x_tof)  # (B*S, 32, 4, 4)
            x_tof = x_tof.reshape(B, S, 32, 16)  # (B, S, 32, 16)
            x_tof = x_tof * mask.unsqueeze(-1).unsqueeze(-1)  # (B, S, 32, 16)
            x_tof = x_tof.transpose(1, 3)  # (B, 16, 32, S)
            x_tof = x_tof.reshape(B * 16, 32, S)  # (B*16, 32, S)
            mask_expanded = mask.unsqueeze(1).repeat(1, 16, 1).reshape(B * 16, S)  # (B*16, S)
            x_tof = self.tof_1d_block_list2[i](x_tof, mask_expanded)  # (B*16, 64, S)
            x_tof = x_tof.reshape(B, 16, 64, S)  # (B, 16, 64, S)
            x_tof = x_tof.transpose(1, 3)  # (B, S, 64, 16)
            x_tof = x_tof.reshape(B * S, 64, 4, 4)  # (B*S, 64, 4, 4)
            x_tof = self.tof_2d_block_list2[i](x_tof)  # (B*S, 128, 2, 2)
            x_tof = x_tof.reshape(B, S, 128, 4)  # (B, S, 128, 4)
            x_tof = x_tof * mask.unsqueeze(-1).unsqueeze(-1)  # (B, S, 128, 4)
            x_tof = x_tof.transpose(1, 3)  # (B, 4, 128, S)
            x_tof = x_tof.reshape(B * 4, 128, S)  # (B*4, 128, S)
            mask_expanded = mask.unsqueeze(1).repeat(1, 4, 1).reshape(B * 4, S)  # (B*4, S)
            x_tof = self.tof_1d_block_list3[i](x_tof, mask_expanded)  # (B*4, 256, S)
            x_tof = x_tof.reshape(B, 4, 256, S)  # (B, 4, 256, S)
            x_tof = x_tof.mean(dim=1)  # (B, 256, S)
            x_tof = x_tof * mask.unsqueeze(1)  # (B, 256, S)
            tof_features.append(x_tof)

        return tof_features
