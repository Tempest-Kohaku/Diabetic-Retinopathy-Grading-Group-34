import torch
import torch.nn as nn
import timm


class ParallelEfficientNetSwin(nn.Module):
    """
    Model architecture used for Hugging Face inference.

    This class must match the architecture used during training exactly.
    The layer names must also match the saved model weights.

    The model combines:
        - EfficientNet features
        - Swin Transformer features
        - Fusion head
        - CORN ordinal output layer
    """

    def __init__(
        self,
        efficientnet_name,
        swin_name,
        num_classes=5,
        fusion_hidden_dim=1024,
        fusion_dropout=0.3,
    ):
        super().__init__()

        self.efficientnet = timm.create_model(
            efficientnet_name,
            pretrained=False,
            num_classes=0,
        )

        self.swin = timm.create_model(
            swin_name,
            pretrained=False,
            num_classes=0,
        )

        eff_dim = getattr(self.efficientnet, "num_features", None)
        swin_dim = getattr(self.swin, "num_features", None)

        if eff_dim is None or swin_dim is None:
            raise ValueError("Could not determine backbone feature dimensions.")

        fusion_dim = eff_dim + swin_dim

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, num_classes - 1),
        )

    @staticmethod
    def to_vector(features):
        if features.ndim == 2:
            return features

        if features.ndim == 3:
            return features.mean(dim=1)

        if features.ndim == 4:
            if (
                features.shape[1] > features.shape[-1]
                and features.shape[1] > features.shape[-2]
            ):
                return features.mean(dim=(2, 3))

            return features.mean(dim=(1, 2))

        raise ValueError(f"Unexpected feature shape: {features.shape}")

    def forward(self, x):
        eff_features = self.to_vector(self.efficientnet(x))
        swin_features = self.to_vector(self.swin(x))

        fused = torch.cat([eff_features, swin_features], dim=1)
        return self.fusion_head(fused)
