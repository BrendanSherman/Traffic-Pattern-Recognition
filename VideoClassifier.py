# VideoClassifier: Handles ViT Initialization, forward pass, and linear classifier
# Streams are handled in paralell, then concatenated before the final layer
# Author - Brendan Sherman 

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms

class VideoClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VideoClassifier, self).__init__()

        # initialize two vit instances (one for each stream)
        self.vit_rgb = models.vit_b_16(pretrained=True)
        self.vit_flow = models.vit_b_16(pretrained=True)

        # Freeze the weights of the early layers
        for param in self.vit_rgb.parameters():
            param.requires_grad = False
        for param in self.vit_flow.parameters():
            param.requires_grad = False

        # Create seperate convolutional layer for flow data
        self.flow_conv = nn.Sequential(
            nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Modify classifier to match combined output size
        num_features_per_stream = 1000
        self.classifier = nn.Linear(num_features_per_stream*2, num_classes)

    # Forward pass carefully treats rgb, flow streams seperately/in paralell
    def forward(self, rgb, flow):
        batch_size, num_frames, rgb_channels, height, width = rgb.size()
        _, num_flow_frames, flow_channels, _, _ = flow.size()

        rgb = rgb.view(batch_size * num_frames, rgb_channels, height, width)
        flow = flow.view(batch_size * num_flow_frames, flow_channels, height, width)

        # Apply custom convolutional layer to flow data for shape compatability
        flow = self.flow_conv(flow)

        rgb_features = self.vit_rgb(rgb)
        flow_features = self.vit_flow(flow)

        rgb_features = rgb_features.view(batch_size, num_frames, -1)
        flow_features = flow_features.view(batch_size, num_flow_frames, -1)

        rgb_features = torch.mean(rgb_features, dim=1)
        flow_features = torch.mean(flow_features, dim=1)

        # Concatenate stream outputs before final classification layer
        combined_streams = torch.cat((rgb_features, flow_features), dim=1)
        out = self.classifier(combined_streams)
        return out

