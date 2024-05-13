import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedKFold

NUM_FOLDS = 5

# Method to calculate optical flow frames given original frame data
def optical_flow(frames):
  flow_frames = []
  for i in range(len(frames) - 1):
    # Calculate optical flow between each pair of consecutive frames
    f1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
    f2 = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow = np.transpose(flow, (2, 0, 1)) # (channels, height, width)
    flow_frames.append(flow)

  return np.array(flow_frames)

# Defining transformations seperately for each stream
def init_transforms():
    rgb_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5)], p=0.5),
        transforms.ToTensor(),
        # normalization params chosen to be optimal for rgb data
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    flow_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.transpose(1, 2, 0)), # (height, width, channels)
        transforms.ToPILImage(),
        # Remove color augmentation for flow data as it is probably unhelpful
        # Keep spatial transformations consistent to maintain spatial correlation
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Adjust normalization params for optical flow data (values range from -1 to 1)
        transforms.Normalize(mean=[0, 0], std=[1, 1])
    ])  
    return rgb_transform, flow_transform



# Class to load video frames and optical flow data for each sample
# Using stratified k-fold CV to combat dataset size and imbalance
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, frames_list, labels_list, indices, rgb_transform=None, flow_transform=None):
        self.frames_list = [frames_list[i] for i in indices]
        self.labels_list = [labels_list[i] for i in indices]
        self.rgb_transform = rgb_transform
        self.flow_transform = flow_transform

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx):
        frames = self.frames_list[idx]
        label = self.labels_list[idx]
        flow_frames = optical_flow(frames)

        # Apply transforms seperately to both streams
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



