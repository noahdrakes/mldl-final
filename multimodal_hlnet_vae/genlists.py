import os
import glob
import random
from pathlib import Path
import numpy as np
def get_video_id(filepath):
    """Extract consistent video ID for both RGB and audio files"""
    filename = os.path.basename(filepath)

    # Handle RGB files (end with __[0-4].npy)
    if filename.endswith(('.npy')):
        base = filename.rsplit('__', 1)[0]  # Split from the right on __ to remove the number or vggish
        return base

    # Fallback - just remove extension
    return filename.split('.')[0]

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
def write_list_files(split_data, aligned_files, outputdir="final_dl/list"): #change path for colab
    """Write list files with audio files repeated to match RGB structure"""
    #os.makedirs(outputdir, exist_ok=True)
    for split_name, video_ids in split_data.items():
        # RGB list - one entry per frame
        rgb_path = os.path.join(outputdir, f'rgb_{split_name}.list')
        with open(rgb_path, 'w') as f:
            for vid_id in video_ids:
                for rgb_file in aligned_files[vid_id]['rgb']:
                    f.write(f"{rgb_file}\n")
        # Audio list - one entry per video (not per frame)
        audio_path = os.path.join(outputdir, f'audio_{split_name}.list')
        with open(audio_path, 'w') as f:
            for vid_id in video_ids:
                audio_file = aligned_files[vid_id]['audio']
                f.write(f"{audio_file}\n")  # Write once only

def find_matching_files():
    """
    Find and align RGB and audio feature files.
    Only includes files that actually exist in the filesystem.
    Returns dict mapping video IDs to their RGB and audio paths
    """
    rgb_path = "final_dl/dl_files/i3d-features/RGB" # change these for colab
    audio_path = "final_dl/list/xx/train"

    # Get all files that actually exist
    rgb_files = [f for f in glob.glob(os.path.join(rgb_path, "*.npy")) if os.path.exists(f)]
    audio_files = [f for f in glob.glob(os.path.join(audio_path, "*.npy")) if os.path.exists(f)]

    # Create mappings that preserve the 5:1 ratio
    rgb_map = {}
    for f in rgb_files:
        vid_id = get_video_id(f)
        if vid_id not in rgb_map:
            rgb_map[vid_id] = []
        rgb_map[vid_id].append(f)

    # Only include audio files that exist
    audio_map = {}
    for f in audio_files:
        if os.path.exists(f):  # Double check existence
            vid_id = get_video_id(f)
            audio_map[vid_id] = f

    # Find common video IDs and verify each RGB group has exactly 5 files
    common_ids = set(rgb_map.keys()) & set(audio_map.keys())
    complete_ids = {vid_id for vid_id in common_ids if len(rgb_map[vid_id]) == 5}

    # Create aligned mapping only for complete groups
    aligned_files = {
        vid_id: {
            'rgb': sorted(rgb_map[vid_id]),  # Sort to maintain consistent ordering
            'audio': audio_map[vid_id],
            'is_normal': '_label_A' in rgb_map[vid_id][0]  # Check first RGB file for label
        }
        for vid_id in complete_ids
    }

    # Print some diagnostic information
    print(f"Total RGB files found: {len(rgb_files)}")
    print(f"Total audio files found: {len(audio_files)}")
    print(f"Video IDs with both RGB and audio: {len(common_ids)}")
    print(f"Complete aligned pairs (5 RGB + 1 audio): {len(aligned_files)}")

    # Print info about incomplete groups
    incomplete_ids = common_ids - complete_ids
    if incomplete_ids:
        print("\nIncomplete groups:")
        for vid_id in incomplete_ids:
            print(f"{vid_id}: {len(rgb_map[vid_id])} RGB files + 1 audio file")

    return aligned_files


aligned_files = find_matching_files()

    # Create train/val splits
split_data = create_splits(aligned_files)

    # Write list files
write_list_files(split_data, aligned_files)
