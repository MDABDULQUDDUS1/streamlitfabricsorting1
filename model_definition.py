"""
Model Definition for Customized DenseNet121
This file contains the exact model architecture used during training
"""

import torch
import torch.nn as nn
import torchvision.models as torch_models


class CustomizedDenseNet121(nn.Module):
    """DenseNet121 with progressive unfreezing strategy"""
    def __init__(self, num_classes=2):
        super(CustomizedDenseNet121, self).__init__()

        # Load pre-trained DenseNet121
        self.backbone = torch_models.densenet121(weights=None)  # weights will be loaded from saved file

        # Initially freeze all feature layers
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # Custom classifier
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        self.unfreeze_schedule = {
            15: 'denseblock4',  # Unfreeze last dense block at epoch 15
            30: 'denseblock3',  # Unfreeze third dense block at epoch 30
        }

    def unfreeze_layer(self, layer_name):
        """Unfreeze specific dense blocks"""
        for name, param in self.backbone.features.named_parameters():
            if layer_name in name:
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


def load_model(model_path, device='cpu'):
    """
    Load the trained model from a .pth file

    Args:
        model_path: Path to the .pth model file
        device: Device to load the model on ('cpu' or 'cuda')

    Returns:
        Loaded model in evaluation mode
    """
    # Create model instance
    model = CustomizedDenseNet121(num_classes=2)

    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Set to evaluation mode
    model.eval()
    model.to(device)

    return model
