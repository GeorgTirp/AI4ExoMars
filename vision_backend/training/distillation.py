from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistillationWrapper(nn.Module):
    def __init__(
        self,
        *,
        teacher_encoder: nn.Module,
        student_encoder: nn.Module,
        teacher_feature_channels: Sequence[int],
        student_feature_channels: Sequence[int],
        feature_indices: Sequence[int],
        feature_weights: Sequence[float] | None = None,
        normalize_features: bool = True,
    ):
        super().__init__()

        if len(teacher_feature_channels) != len(student_feature_channels):
            raise ValueError("teacher_feature_channels and student_feature_channels must match in length.")
        if len(teacher_feature_channels) != len(feature_indices):
            raise ValueError("feature_indices must align with the teacher/student channel lists.")

        self.teacher = teacher_encoder
        self.student = student_encoder
        self.feature_indices = list(feature_indices)
        self.feature_weights = (
            list(feature_weights)
            if feature_weights is not None
            else [1.0] * len(self.feature_indices)
        )
        if len(self.feature_weights) != len(self.feature_indices):
            raise ValueError("feature_weights must match feature_indices in length.")

        self.normalize_features = normalize_features
        self.projectors = nn.ModuleList(
            [
                nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False)
                for teacher_channels, student_channels in zip(
                    teacher_feature_channels,
                    student_feature_channels,
                )
            ]
        )

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def _normalize(self, feature: torch.Tensor) -> torch.Tensor:
        if not self.normalize_features:
            return feature
        return F.normalize(feature, dim=1)

    def forward(
        self,
        local_x: torch.Tensor,
        context_x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            teacher_features = self.teacher(local_x, context_x)

        student_features = self.student(local_x, context_x)

        total_loss = local_x.new_tensor(0.0)
        losses: dict[str, torch.Tensor] = {}

        for loss_index, (feature_index, weight, projector) in enumerate(
            zip(self.feature_indices, self.feature_weights, self.projectors)
        ):
            teacher_feature = teacher_features[feature_index].detach()
            student_feature = projector(student_features[feature_index])

            if student_feature.shape[2:] != teacher_feature.shape[2:]:
                student_feature = F.interpolate(
                    student_feature,
                    size=teacher_feature.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            teacher_feature = self._normalize(teacher_feature)
            student_feature = self._normalize(student_feature)

            feature_loss = F.mse_loss(student_feature, teacher_feature)
            total_loss = total_loss + weight * feature_loss
            losses[f"feature_{loss_index}_loss"] = feature_loss

        losses["loss"] = total_loss
        return losses
