import os
import glob
import random
from pathlib import Path
import numpy as np


def random_extract(feat, t_max):
   r = np.random.randint(len(feat)-t_max)
   return feat[r:r+t_max]

def uniform_extract(feat, t_max):
   r = np.linspace(0, len(feat)-1, t_max, dtype=np.uint16)
   return feat[r, :]

def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
       return np.pad(feat, ((0, min_len-np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
       return feat

def process_feat(feat, length, is_random=True):
    if len(feat) > length:
        if is_random:
            return random_extract(feat, length)
        else:
            return uniform_extract(feat, length)
    else:
        return pad(feat, length)


def collate_fn_audio(batch):
    """
    Collate function for a batch of spectrograms.
    Each spectrogram is of shape [time_steps, num_mels].
    This function pads each spectrogram along the time dimension 
    so that all have the same time_steps dimension.
    """
    # `batch` is a list of spectrograms: each is numpy array [T, num_mels]
    # Find the longest time dimension in the batch
    max_length = max(feat.shape[0] for feat in batch)

    # Pad each spectrogram to max_length
    padded_feats = []
    for feat in batch:
        t, m = feat.shape
        # Create a zero array [max_length, num_mels]
        padded = np.zeros((max_length, m), dtype=np.float32)
        padded[:t] = feat
        padded_feats.append(padded)

    # Convert the list of numpy arrays into a single tensor
    feats_batch = torch.tensor(padded_feats, dtype=torch.float32)

    return feats_batch


def get_video_id(filepath):
    """Extract video ID from filepath based on common prefix before _label
    e.g., "/path/to/video123_label_A.npy" -> "video123"
    """
    filename = os.path.basename(filepath)
    if '_label' in filename:
        return filename.split('_label')[0]
    return filename.split('.')[0]

def find_matching_files():
    """
    Find and align RGB and audio feature files.
    Returns dict mapping video IDs to their RGB and audio paths
    """
    rgb_path = "/export/fs05/bodoom1/dl_proj/final_dl/dl_files/i3d-features/RGB"
    audio_path = "/export/fs05/bodoom1/dl_proj/final_dl/list/xx/train"

    # Get all files
    rgb_files = glob.glob(os.path.join(rgb_path, "*.npy"))
    audio_files = glob.glob(os.path.join(audio_path, "*.npy"))

    # Create mappings that preserve the 5:1 ratio
    rgb_map = {}
    for f in rgb_files:
        vid_id = get_video_id(f)
        if vid_id not in rgb_map:
            rgb_map[vid_id] = []
        rgb_map[vid_id].append(f)

    audio_map = {get_video_id(f): f for f in audio_files}

    # Find common video IDs
    common_ids = set(rgb_map.keys()) & set(audio_map.keys())

    # Create aligned mapping
    aligned_files = {
        vid_id: {
            'rgb': sorted(rgb_map[vid_id]),  # Sort to maintain consistent ordering
            'audio': audio_map[vid_id],
            'is_normal': '_label_A' in rgb_map[vid_id][0]  # Check first RGB file for label
        }
        for vid_id in common_ids
    }

    print(f"Found {len(aligned_files)} aligned RGB-Audio pairs")
    return aligned_files

def create_splits(aligned_files, train_ratio=0.8, seed=42):
    """Split the video IDs first, then we'll expand to files in write_list_files"""
    random.seed(seed)
    video_ids = list(aligned_files.keys())
    train_size = int(len(video_ids) * train_ratio)
    train_ids = random.sample(video_ids, train_size)
    val_ids = [vid for vid in video_ids if vid not in train_ids]

    return {
        'train': train_ids,
        'val': val_ids
    }

def write_list_files(split_data, aligned_files, output_dir="/content/final_dl/list"):
    """Write list files with audio files repeated to match RGB structure"""
    os.makedirs(output_dir, exist_ok=True)

    for split_name, video_ids in split_data.items():
        # RGB list - one entry per frame
        rgb_path = os.path.join(output_dir, f'rgb_{split_name}.list')
        with open(rgb_path, 'w') as f:
            for vid_id in video_ids:
                for rgb_file in aligned_files[vid_id]['rgb']:
                    f.write(f"{rgb_file}\n")

        # Audio list - one entry per video (not per frame)
        audio_path = os.path.join(output_dir, f'audio_{split_name}.list')
        with open(audio_path, 'w') as f:
            for vid_id in video_ids:
                audio_file = aligned_files[vid_id]['audio']
                f.write(f"{audio_file}\n")  # Write once only
