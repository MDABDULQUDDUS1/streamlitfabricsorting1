"""
Model Definition for Customized DenseNet121
This file contains the exact model architecture used during training.
"""

import torch
import torch.nn as nn
import torchvision.models as torch_models
import os

class CustomizedDenseNet121(nn.Module):
    """DenseNet121 with progressive unfreezing strategy"""
    def __init__(self, num_classes=2):
        super(CustomizedDenseNet121, self).__init__()

        # Load pre-trained DenseNet121 (weights will be loaded from saved file)
        self.backbone = torch_models.densenet121(weights=None)

        # Freeze all feature layers initially
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # Custom classifier head
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        # Optional unfreeze schedule (for training only)
        self.unfreeze_schedule = {
            15: 'denseblock4',
            30: 'denseblock3',
        }

    def unfreeze_layer(self, layer_name):
        """Unfreeze specific dense blocks"""
        for name, param in self.backbone.features.named_parameters():
            if layer_name in name:
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


def merge_model_parts(part1_path, part2_path, merged_path="merged_model.pth"):
    """Merge split model files into a single .pth file"""
    with open(merged_path, "wb") as outfile:
        for part in [part1_path, part2_path]:
            with open(part, "rb") as infile:
                outfile.write(infile.read())
    return merged_path


def load_model(model_dir=".", device='cpu'):
    """
    Load the trained model from split .pth files.

    Args:
        model_dir: Directory containing the split model parts
        device: 'cpu' or 'cuda'

    Returns:
        Loaded model ready for inference
    """
    part1 = os.path.join(model_dir, "densenet121_part1.pth")
    part2 = os.path.join(model_dir, "densenet121_part2.pth")
    merged = os.path.join(model_dir, "final_merged_model.pth")

    # Merge if the merged file doesn’t exist yet
    if not os.path.exists(merged):
        if not (os.path.exists(part1) and os.path.exists(part2)):
            raise FileNotFoundError("Missing model part files.")
        merge_model_parts(part1, part2, merged)

    # Load model
    model = CustomizedDenseNet121(num_classes=2)
    state_dict = torch.load(merged, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model
