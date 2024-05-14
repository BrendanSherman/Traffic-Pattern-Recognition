# Methods for importing, parsing, and labelling of video dataset
# Part of Larger Traffic Pattern Recognition Project
# Author - Brendan Sherman (shermabk@bc.edu)

from io import StringIO
import cv2
import numpy as np
import pandas as pd

VID_HEIGHT = 224
VID_WIDTH = 224

# Load dataset attributes from csv
def load_dataset(path_str):
    path = path_str
    df= pd.read_csv(path) 
    
    # Cleaning csv file (whitespace, tab seperation)
    file_path = path 
    with open(file_path, 'r') as file:
        lines = file.readlines()
    lines[0] = lines[0].replace(',', '\t')
    modified_file_content = ''.join(lines)

    df = pd.read_csv(StringIO(modified_file_content), delimiter='\t')
    df = df.rename(columns={
        ' date(yyyymmdd)': 'date(yyyymmdd)',
        ' timestamp': 'timestamp',
        ' direction': 'direction',
        ' day/night': 'day/night',
        ' weather': 'weather',
        ' start frame': 'start frame',
        ' number of frames': 'number of frames',
        ' class': 'class',
        ' notes': 'notes'
    })

    return df

# Using csv data, label each video file with its classification
def label_samples(df):
    df = df.sample(frac=1)
    frames_list = []
    labels_list = []

    vid_height = VID_HEIGHT
    vid_width = VID_WIDTH
    num_channels = 3

    for _, row in df.iterrows():
        start_frame = row['start frame']
        num_frames = 40
        
        # Load each video file and initialize ndarray for frame data
        # shape = (num_frames, h, w, num_channels)
        cap = cv2.VideoCapture('/archive 2/video/' + row['# filename'] + '.avi')
        frames = np.empty((num_frames, vid_height, vid_width, num_channels), dtype=np.uint8)

        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
            ret, frame = cap.read()
            if not ret:
                break

            # Resize each frame and append to ndarray         
            frame = cv2.resize(frame, (vid_height, vid_width))
            frames[i] = frame

        cap.release()

        # Each 'frames' ndarray in frames_list has corresponding label in labels_list
        frames_list.append(frames)
        label = row['class']
        labels_list.append(label)

    return (frames_list, labels_list)

# Use labels list to count classifications in dataset 
# There is a clear class imbalance, so stratification might be necessary
def num_counts(labels_list):
    heavy_counts, medium_counts, light_counts = 0,0,0
    for l in labels_list:
        if l == 'heavy':
            heavy_counts += 1
        elif l == 'medium':
            medium_counts += 1
        else:
            light_counts += 1

    print("# 'heavy' Samples: ", heavy_counts)
    print("# 'medium' Samples: ", medium_counts)
    print("# 'light' Samples: ", light_counts)        
    return (heavy_counts, medium_counts, light_counts)


