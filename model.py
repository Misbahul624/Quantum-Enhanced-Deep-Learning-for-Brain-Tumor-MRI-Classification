"""
Model factory for the Brain Tumor MRI classification project.
Provides three primary models as an ensemble:
  1. ConvNeXt-Small  (IN-22k → IN-1k → fine-tune)   – best single model candidate
  2. EfficientNetV2-M (IN-21k → IN-1k → fine-tune)   – strong complementary
  3. Swin-Base (IN-22k → IN-1k → fine-tune)           – hierarchical transformer
"""
import torch
import torch.nn as nn
import timm

NUM_CLASSES = 4


def build_model(arch: str, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """
    Build a timm model with a custom two-layer classification head that includes
    dropout for regularisation. Supports three backbones:
      - 'convnext_small'
      - 'efficientnetv2_m'
      - 'swin_base'
    """
    arch_map = {
        'convnext_small': 'convnext_small.fb_in22k_ft_in1k',
        'efficientnetv2_m': 'tf_efficientnetv2_m.in21k_ft_in1k',
        'swin_base': 'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
    }

    model_name = arch_map.get(arch, arch)
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)  # feature extractor

    num_features = model.num_features
    model.head_drop = nn.Dropout(p=dropout)
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.LayerNorm(512),   # LayerNorm works with batch_size=1; BN does not
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(512, NUM_CLASSES)
    )

    def forward_with_head(self, x):
        features = self.forward_features(x)
        # Pool: (B, H, W, C) for Swin, (B, C, H, W) for ConvNeXt/EfficientNet
        if features.dim() == 4:
            if features.shape[1] in (num_features,):   # channel-first
                features = features.mean(dim=[2, 3])
            else:                                        # channel-last (Swin)
                features = features.mean(dim=[1, 2])
        elif features.dim() == 3:                        # Swin with cls token
            features = features.mean(dim=1)
        features = self.head_drop(features)
        return self.classifier(features)

    import types
    model.forward = types.MethodType(forward_with_head, model)
    return model


def build_ensemble(archs: list[str], pretrained: bool = True, dropout: float = 0.3) -> list[nn.Module]:
    return [build_model(arch, pretrained=pretrained, dropout=dropout) for arch in archs]


class EnsembleModel(nn.Module):
    """
    Wraps a list of models and averages their softmax/logit predictions.
    Used for inference only after each member is trained independently.
    """
    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([m(x) for m in self.models], dim=0)
        return logits.mean(dim=0)
