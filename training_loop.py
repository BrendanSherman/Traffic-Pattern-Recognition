# Main training loop definition
# Note - Commented out some lines meant for use with GPU/CUDA
# Author - Brendan Sherman  

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from import_data import *
from preprocessing import * 
from VideoClassifier import VideoClassifier
from VideoDataset import VideoDataset

ACCUMULATION_STEPS = 4
BATCH_SIZE = 2
DATA_PATH = '' #TODO update
NUM_CLASSES = 3
NUM_FOLDS = 5
NUM_EPOCHS = 5
TOTAL_SAMPLES = 165 + 45 + 44
NUM_LIGHT = 165
NUM_MED = 45
NUM_HEAVY = 44

def init_training():
    df = load_dataset(DATA_PATH)
    frames_list, labels_list = label_samples(df)
    rgb_transform, flow_transform = init_transforms()
    model = VideoClassifier(NUM_CLASSES)
    #model = model.to(device)
    # Define weighted loss using inverse of class frequencies
    class_weights = torch.tensor([TOTAL_SAMPLES / NUM_LIGHT, TOTAL_SAMPLES / NUM_MED, TOTAL_SAMPLES / NUM_HEAVY]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights) # Weighted loss function due to imbalance
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True) # LR Scheduler
    # Integer class mappings for CV
    label_map = {'light': 0, 'medium': 1, 'heavy': 2}

    return model, label_map, scheduler, criterion, optimizer, frames_list, labels_list

    # Training loop
    # With stratified k-fold CV and transformations defined above
    # Utilizes gradient accumulation to simulate larger batch size
def trainingloop():
    model, label_map, scheduler, criterion, optimizer, frames_list, labels_list = init_training()
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(skf.split(frames_list, labels_list), 1):
        train_dataset = VideoDataset(frames_list, labels_list, train_indices, rgb_transform=rgb_transform, flow_transform=flow_transform)
        val_dataset = VideoDataset(frames_list, labels_list, val_indices, rgb_transform=rgb_transform, flow_transform=flow_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()
            for i, data in enumerate(train_loader, 0):
                frames, flow_frames, labels = data
                #frames = frames.to(device)
                #flow_frames = flow_frames.to(device)
                labels = [label_map[label] for label in labels]
                #labels = torch.tensor(labels).to(device)
                #outputs = model(frames, flow_frames)
                loss = criterion(outputs, labels)
                train_loss += loss.item()
                loss = loss / ACCUMULATION_STEPS
                loss.backward()

                if (i + 1) % ACCUMULATION_STEPS == 0: # Gradient accumulation check
                    optimizer.step()
                    optimizer.zero_grad()

            train_loss /= len(train_loader)

            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for data in train_loader:
                    frames, flow_frames, labels = data
                    #frames = frames.to(device)
                    #flow_frames = flow_frames.to(device)
                    labels = [label_map[label] for label in labels]
                    #labels = torch.tensor(labels).to(device)
                    outputs = model(frames, flow_frames)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                train_accuracy = correct / total

            print(f"Fold {fold}, Epoch [{epoch+1}/{NUM_EPOCHS}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, ", end ='')

            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for data in val_loader:
                    frames, flow_frames, labels = data
                    #frames = frames.to(device) 
                    #flow_frames = flow_frames.to(device)
                    labels = [label_map[label] for label in labels]
                    #labels = torch.tensor(labels).to(device)
                    outputs = model(frames, flow_frames)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accuracy = correct / total
                print(f"Validation Accuracy: {accuracy:.4f}")

            scheduler.step(accuracy) # Update LR Scheduler with validation accuracy

