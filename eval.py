import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
import argparse
import os
from model import Model
from datetime import datetime

class Args:
    def __init__(self, modality='RGB'):
        self.modality = modality
        # Base paths
        self.test_rgb_list = './final_dl/list/rgb_test.list'
        self.test_audio_list = './final_dl/list/audio_test.list'

        # Model parameters
        self.batch_size = 5  # Small batch size for testing
        self.workers = 1
        self.model_name = 'wsanodet'
        self.num_classes = 1
        self.max_seqlen = 200

        # Set feature size based on modality
        if modality == 'RGB':
            self.feature_size = 1024
        elif modality == 'MIX2':
            self.feature_size = 1152
        elif modality == 'AUDIO':
            self.feature_size = 128
        else:
            raise ValueError(f"Unsupported modality: {modality}")

class Dataset(data.Dataset):
    def __init__(self, args):
        self.modality = args.modality
        self.max_seqlen = args.max_seqlen

        if self.modality == 'RGB':
            self.list_file = args.test_rgb_list
            self.list = [line.strip() for line in open(self.list_file)]
        elif self.modality == 'AUDIO':
            self.list_file = args.test_audio_list
            self.list = [line.strip() for line in open(self.list_file)]
        elif self.modality == 'MIX2':
            self.rgb_list = [line.strip() for line in open(args.test_rgb_list)]
            self.audio_list = [line.strip() for line in open(args.test_audio_list)]
            self.list = self.rgb_list

    def __getitem__(self, index):
        if self.modality == 'RGB':
            file_path = self.list[index].strip()
            features = np.array(np.load(file_path), dtype=np.float32)
            label = 0.0 if '_label_A' in file_path else 1.0
        elif self.modality == 'AUDIO':
            file_path = self.list[index].strip()
            features = np.array(np.load(file_path), dtype=np.float32)
            label = 0.0 if '_label_A' in file_path else 1.0
        elif self.modality == 'MIX2':  # MIX2
            file_path1 = self.list[index].strip()
            features1 = np.array(np.load(file_path1), dtype=np.float32)
            label = 0.0 if '_label_A' in file_path1 else 1.0

            audio_index = index // 5
            file_path2 = self.audio_list[audio_index].strip()
            features2 = np.array(np.load(file_path2), dtype=np.float32)

            features = np.concatenate((features1, features2), axis=1)

        features = self.process_feat(features, self.max_seqlen)
        return features, label

    def __len__(self):
        return len(self.list)

    def process_feat(self, feat, length):
        if len(feat) > length:
            r = np.linspace(0, len(feat)-1, length, dtype=np.uint16)
            return feat[r, :]
        else:
            return np.pad(feat, ((0, length-np.shape(feat)[0]), (0, 0)),
                         mode='constant', constant_values=0)

def evaluate_model(model_path, modality='RGB', output_dir='evaluation_results'):

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args = Args(modality=modality)
    model = Model(args).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_dataset = Dataset(args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    print(f"Created test loader with {len(test_dataset)} samples")

    video_gt = []
    video_pred = []
    video_pred2 = []
    current_video_preds = []
    current_video_preds2 = []

    print("Starting evaluation...")
    with torch.no_grad():
        for i, (input, label) in enumerate(test_loader):
            input = input.to(device)
            logits, logits2 = model(inputs=input, seq_len=None)

            logits = torch.squeeze(logits)
            sig = torch.sigmoid(logits)
            sample_preds = torch.mean(sig, dim=1).cpu().numpy()
            current_video_preds.extend(sample_preds)

            logits2 = torch.squeeze(logits2)
            sig2 = torch.sigmoid(logits2)
            sample_preds2 = torch.mean(sig2, dim=1).cpu().numpy()
            current_video_preds2.extend(sample_preds2)

            # Aggregate predictions every 5 frames
            if (i + 1) % 5 == 0:
                video_pred.append(np.mean(current_video_preds[-5:]))
                video_pred2.append(np.mean(current_video_preds2[-5:]))
                video_gt.append(label[0].item())
                current_video_preds = []
                current_video_preds2 = []

    # Convert to numpy arrays
    video_gt = np.array(video_gt)
    print(f"Number of test samples processed: {len(video_gt)}")
    video_pred = np.array(video_pred)
    video_pred2 = np.array(video_pred2)

    # Calculate metrics
    metrics = {
        'AP': average_precision_score(video_gt, video_pred) * 100,
        'AP_online': average_precision_score(video_gt, video_pred2) * 100
    }

    # Calculate PR curves
    precision, recall, _ = precision_recall_curve(video_gt, video_pred)
    precision2, recall2, _ = precision_recall_curve(video_gt, video_pred2)
    metrics['PR_AUC'] = auc(recall, precision) * 100
    metrics['PR_AUC_online'] = auc(recall2, precision2) * 100

    # Save metrics to file
    metrics_path = os.path.join(output_dir, f'metrics_{modality}.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Modality: {modality}\n\n")
        f.write("Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.2f}%\n")

    print("\nEvaluation complete!")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--modality', type=str, choices=['RGB', 'MIX2', 'AUDIO'],
                      default='RGB', help='Model modality (RGB, AUDIO, or MIX2)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save results')

    args = parser.parse_args()

    metrics = evaluate_model(
        model_path=args.model_path,
        modality=args.modality,
        output_dir=args.output_dir
    )

    # Print metrics
    print("\nEvaluation Results:")
    print(f"Modality: {args.modality}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}%")
