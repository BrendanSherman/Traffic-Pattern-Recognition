# Class to load video frames and optical flow data for each sample
# Uses stratified k-fold CV to combat dataset size and imbalance
# Author - Brendan Sherman

import torch
import torch.nn as nn
import torchvision.models as models
from preprocessing import optical_flow, init_transforms

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, frames_list, labels_list, indices, rgb_transform=None, flow_transform=None):
        self.frames_list = [frames_list[i] for i in indices]
        self.labels_list = [labels_list[i] for i in indices]
        rgb_tr, flow_tr = init_transforms()
        self.rgb_transform = rgb_tr
        self.flow_transform = flow_tr

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx):
        frames = self.frames_list[idx]
        label = self.labels_list[idx]
        flow_frames = optical_flow(frames)

        # Apply correct transforms to both streams
        if self.rgb_transform:
          transformed_frames = []
          for frame in frames:
            transformed_frame = self.rgb_transform(frame)
            transformed_frames.append(transformed_frame)
          frames = torch.stack(transformed_frames)

        if self.flow_transform:
          transformed_flow_frames = []
          for flow_frame in flow_frames:
            transformed_flow_frame = self.flow_transform(flow_frame)
            transformed_flow_frames.append(transformed_flow_frame)
          flow_frames = torch.stack(transformed_flow_frames)

        return frames, flow_frames, label