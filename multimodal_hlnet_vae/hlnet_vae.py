import torch.utils.data as data
import torch.optim as optim
import numpy as np
import os
import glob
import random
from pathlib import Path
import torch
from torch import nn
import torch
from datetime import datetime
from model import Model
from layers import GraphAttentionLayer, linear, GraphConvolution, SimilarityAdj
from sklearn.metrics import auc, precision_recall_curve
#from utils import *
from VAE import *
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class Args:
    def __init__(self):
        self.modality = 'MIX2'
        # these paths are changed for me using it on the cluster
        self.rgb_list = './final_dl/list/rgb.list'
        self.flow_list = './final_dl/list/flow.list'
        self.audio_list = './final_dl/list/audio.list'

        # Train paths
        self.train_rgb_list = './final_dl/list/rgb_train.list'
        self.train_flow_list = './final_dl/list/flow_train.list'
        self.train_audio_list = './final_dl/list/audio_train.list'

        # Val paths
        self.val_rgb_list = './final_dl/list/rgb_val.list'
        self.val_flow_list = './final_dl/list/flow_val.list'
        self.val_audio_list = './final_dl/list/audio_val.list'

        # Test paths
        self.test_rgb_list = './final_dl/list/rgb_test.list'
        self.test_flow_list = './final_dl/list/flow_test.list'
        self.test_audio_list = './final_dl/list/audio_test.list'

        self.gt = './final_dl/list/gt.npy'
        self.gpus = 1
        self.lr = 0.0001
        self.batch_size = 128
        self.workers = 1  # Reduced from 4 to avoid memory issues
        self.model_name = 'wsanodet'
        self.pretrained_ckpt = None
        self.feature_size = 1152  # 1024 + 128
        self.num_classes = 1
        self.dataset_name = 'XD-Violence'
        self.max_seqlen = 200
        self.max_epoch = 50

args = Args()

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


class Dataset(data.Dataset):
    def __init__(self, args, transform=None, mode='train'):
        self.modality = args.modality
        self.max_seqlen = args.max_seqlen
        self.transform = transform
        self.test_mode = (mode == 'test')

        # Set appropriate file lists based on mode
        if mode == 'test':
            self.rgb_list_file = args.test_rgb_list
            self.flow_list_file = args.test_flow_list
            self.audio_list_file = args.test_audio_list
        elif mode == 'val':
            self.rgb_list_file = args.val_rgb_list
            self.flow_list_file = args.val_flow_list
            self.audio_list_file = args.val_audio_list
        else:  # train
            self.rgb_list_file = args.train_rgb_list
            self.flow_list_file = args.train_flow_list
            self.audio_list_file = args.train_audio_list

        self._parse_list()

    def _parse_list(self):
        """Parse file lists - assumes lists are already properly aligned"""
        if self.modality == 'MIX2':
            self.list = [line.strip() for line in open(self.rgb_list_file)]
            self.audio_list = [line.strip() for line in open(self.audio_list_file)]
        elif self.modality == 'AUDIO':
            self.list = [line.strip() for line in open(self.audio_list_file)]
        elif self.modality == 'RGB':
            self.list = [line.strip() for line in open(self.rgb_list_file)]
        elif self.modality == 'FLOW':
            self.list = [line.strip() for line in open(self.flow_list_file)]

    def __getitem__(self, index):
        if self.modality in ['RGB', 'FLOW', 'AUDIO']:
            file_path = self.list[index].strip()
            features = np.array(np.load(file_path), dtype=np.float32)
            label = 0.0 if '_label_A' in file_path else 1.0
        elif self.modality == 'MIX2':
            # Load RGB features
            file_path1 = self.list[index].strip()
            features1 = np.array(np.load(file_path1), dtype=np.float32)
            label = 0.0 if '_label_A' in file_path1 else 1.0

            # Load corresponding audio features
            audio_index = index // 5
            file_path2 = self.audio_list[audio_index].strip()
            features2 = np.array(np.load(file_path2), dtype=np.float32)

            features = np.concatenate((features1, features2), axis=1)

        if self.transform is not None:
            features = self.transform(features)

        features = process_feat(features, self.max_seqlen, is_random=not self.test_mode)
        return features, label

    def __len__(self):
        return len(self.list)


def CLAS(logits, label, seq_len, criterion, device, is_topk=True):
    logits = logits.squeeze()
    instance_logits = torch.zeros(0).to(device)  # tensor([])
    for i in range(logits.shape[0]):
        if is_topk:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i]//16+1), largest=True)
            tmp = torch.mean(tmp).view(1)
        else:
            tmp = torch.mean(logits[i, :seq_len[i]]).view(1)
        instance_logits = torch.cat((instance_logits, tmp))

    instance_logits = torch.sigmoid(instance_logits)

    clsloss = criterion(instance_logits, label)
    return clsloss


def CENTROPY(logits, logits2, seq_len, device):
    instance_logits = torch.tensor(0).to(device)  # tensor([])
    for i in range(logits.shape[0]):
        tmp1 = torch.sigmoid(logits[i, :seq_len[i]]).squeeze()
        tmp2 = torch.sigmoid(logits2[i, :seq_len[i]]).squeeze()
        loss = torch.mean(-tmp1.detach() * torch.log(tmp2))
        instance_logits = instance_logits + loss
    instance_logits = instance_logits/logits.shape[0]
    return instance_logits


def validate_epoch(val_loader, hlnet, vae, criterion, device, is_topk,
                  HLNET_LOSS_WEIGHT, RECON_LOSS_WEIGHT):
    """Run validation for one epoch"""
    hlnet.eval()
    vae.eval()
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for input, label in val_loader:
            inputcpy = input.float().to(device)
            seq_len = torch.sum(torch.max(torch.abs(input), dim=2)[0]>0, 1)
            input = input[:, :torch.max(seq_len), :]
            input, label = input.float().to(device), label.float().to(device)

            logits, logits2 = hlnet(input, seq_len)
            clsloss = CLAS(logits, label, seq_len, criterion, device, is_topk)
            clsloss2 = CLAS(logits2, label, seq_len, criterion, device, is_topk)
            croloss = CENTROPY(logits, logits2, seq_len, device)

            recon_data, mu, logvar = vae(inputcpy)
            recon_criterion = torch.nn.MSELoss(reduction='mean')
            recon_loss = recon_criterion(recon_data, inputcpy)

            total_loss += (HLNET_LOSS_WEIGHT * (clsloss + clsloss2 + 5*croloss) +
                         RECON_LOSS_WEIGHT * recon_loss).item()
            batch_count += 1

    return total_loss / batch_count

def test_hl_vae(dataloader, model, device):
    """
    Test function for HL-Net that evaluates video-level predictions
    Returns PR-AUC scores for both offline and online predictions
    """
    model.eval()
    video_gt = []
    video_pred = []
    video_pred2 = []

    current_video_preds = []
    current_video_preds2 = []

    with torch.no_grad():
        for i, (input, label) in enumerate(dataloader):
            # Process input
            input = input.to(device)
            logits, logits2 = model(inputs=input, seq_len=None)

            # Process predictions - average across time dimension first
            logits = torch.squeeze(logits)  # Remove batch dim if batch_size=1
            sig = torch.sigmoid(logits)
            # Take mean across time dimension for each sample
            sample_preds = torch.mean(sig, dim=1).cpu().numpy()
            current_video_preds.extend(sample_preds)

            logits2 = torch.squeeze(logits2)
            sig2 = torch.sigmoid(logits2)
            sample_preds2 = torch.mean(sig2, dim=1).cpu().numpy()
            current_video_preds2.extend(sample_preds2)

            # Every 5 frames, compute video-level prediction
            if (i + 1) % 5 == 0:
                # Take mean of the 5 frame predictions for this video
                video_pred.append(np.mean(current_video_preds[-5:]))
                video_pred2.append(np.mean(current_video_preds2[-5:]))
                # Only take one label per video (they're all the same)
                video_gt.append(label[0].item())
                # Reset for next video
                current_video_preds = []
                current_video_preds2 = []

    # Convert to numpy arrays
    video_gt = np.array(video_gt)
    video_pred = np.array(video_pred)
    video_pred2 = np.array(video_pred2)

    # Print shapes for debugging
    print(f"Shapes - GT: {video_gt.shape}, Pred: {video_pred.shape}, Pred2: {video_pred2.shape}")

    # Calculate metrics
    precision, recall, _ = precision_recall_curve(video_gt, video_pred)
    pr_auc = auc(recall, precision)

    precision2, recall2, _ = precision_recall_curve(video_gt, video_pred2)
    pr_auc2 = auc(recall2, precision2)

    return pr_auc, pr_auc2

def train_hlnet_vae(train_loader, hlnet, vae, optimizer, scheduler, criterion,
                    device, is_topk, HLNET_LOSS_WEIGHT, RECON_LOSS_WEIGHT, val_loader=None):
    """Training function with loss tracking"""
    hlnet.train()
    vae.eval()
    epoch_loss = 0.0
    batch_count = 0

    for i, (input, label) in enumerate(train_loader):
        inputcpy = input.float().to(device)
        seq_len = torch.sum(torch.max(torch.abs(input), dim=2)[0]>0, 1)
        input = input[:, :torch.max(seq_len), :]
        input, label = input.float().to(device), label.float().to(device)

        logits, logits2 = hlnet(input, seq_len)
        clsloss = CLAS(logits, label, seq_len, criterion, device, is_topk)
        clsloss2 = CLAS(logits2, label, seq_len, criterion, device, is_topk)
        croloss = CENTROPY(logits, logits2, seq_len, device)

        with torch.inference_mode():
            recon_data, mu, logvar = vae(inputcpy)
            recon_criterion = torch.nn.MSELoss(reduction='mean')
            recon_loss = recon_criterion(recon_data, inputcpy)

        total_loss = HLNET_LOSS_WEIGHT * (clsloss + clsloss2 + 5*croloss) + RECON_LOSS_WEIGHT * recon_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        batch_count += 1

        if (i + 1) % 100 == 0:
            print(f"Step {i+1}: Training Loss: {total_loss.item():.4f}")

    avg_epoch_loss = epoch_loss / batch_count

    # Calculate validation loss if provided
    val_epoch_loss = None
    if val_loader is not None:
        val_epoch_loss = validate_epoch(val_loader, hlnet, vae, criterion,
                                      device, is_topk, HLNET_LOSS_WEIGHT, RECON_LOSS_WEIGHT)

    return hlnet, avg_epoch_loss, val_epoch_loss


def create_data_loaders(args):
    """
    Create train, validation and test data loaders
    """
    print("Creating data loaders...")

    # Create train loader
    train_dataset = Dataset(args, mode='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    print(f"Train loader created with {len(train_dataset)} samples")

    # Create validation loader
    val_dataset = Dataset(args, mode='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=args.workers,
        pin_memory=True
    )
    print(f"Validation loader created with {len(val_dataset)} samples")

    # Create test loader with smaller batch size as per original code
    test_dataset = Dataset(args, mode='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=5,  # Using smaller batch size for testing
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    print(f"Test loader created with {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader

def plot_training_history(train_losses, val_losses, save_path='hl-vae-training_history.png'):
    """
    Plot training and validation losses over epochs.

    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    epochs = range(len(train_losses))

    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')

    plt.title('HL-Net + VAE Multimodal Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    plt.close()


def main():
    # Initialize models and data loaders as before
    args_vae = Args()
    args_vae.feature_size = 1152
    args_vae.batch_size = 64
    args_vae.modality = 'MIX2'
    args_vae.max_epoch = 500
    args_vae.lr = 0.0005

    vae_model = VAE(latent_dim=64, input_dim=args_vae.feature_size, seq_len=200)
    dir = "./mm_best_trained_vae.pkl"
    vae_model.load_state_dict(torch.load(dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_model = vae_model.cuda()
    vae_model.eval()

    HL_NET_LOSS_weight = .8
    RECON_LOSS_weight = .2

    args = Args()
    args.feature_size = 1152
    args.batch_size = 128
    args.modality = 'MIX2'
    args.max_seqlen = 200
    args.workers = 1

    train_loader, val_loader, test_loader = create_data_loaders(args)
    model = Model(args).to(device)

    approximator_param = list(map(id, model.approximator.parameters()))
    approximator_param += list(map(id, model.conv1d_approximator.parameters()))
    base_param = filter(lambda p: id(p) not in approximator_param, model.parameters())

    optimizer = optim.Adam([
        {'params': base_param},
        {'params': model.approximator.parameters(), 'lr': args.lr / 2},
        {'params': model.conv1d_approximator.parameters(), 'lr': args.lr / 2},
    ], lr=args.lr, weight_decay=0.000)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
    criterion = torch.nn.BCELoss()
    is_topk = True

    # Lists to store training history
    train_losses = []
    val_losses = []

    for epoch in range(args.max_epoch):
        print(f"\nEpoch {epoch+1}/{args.max_epoch}")
        st = time.time()

        # Training step
        model, train_loss, val_loss = train_hlnet_vae(
            train_loader=train_loader,
            hlnet=model,
            vae=vae_model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            is_topk=is_topk,
            HLNET_LOSS_WEIGHT=HL_NET_LOSS_weight,
            RECON_LOSS_WEIGHT=RECON_LOSS_weight,
            val_loader=val_loader
        )

        # Store losses
        train_losses.append(train_loss)
        if val_loss is not None:
            val_losses.append(val_loss)

        # Step scheduler at epoch level
        scheduler.step()

        # Save checkpoint
        if epoch % 2 == 0 and epoch > 0:
            torch.save(model.state_dict(), f'./hlnet_saves/{args.model_name}{epoch}.pth')

        # Evaluate
        pr_auc, pr_auc_online = test_hl_vae(test_loader, model, device)

        print(f'Epoch {epoch}/{args.max_epoch}:')
        print(f'Train Loss: {train_loss:.4f}')
        if val_loss is not None:
            print(f'Validation Loss: {val_loss:.4f}')
        print(f'Video-level PR-AUC:')
        print(f'  Offline: {pr_auc:.4f}')
        print(f'  Online: {pr_auc_online:.4f}')
        print(f'Epoch time: {time.time() - st:.2f}s')

    # Save final model
    torch.save(model.state_dict(), f'./hlnet_saves/final/{args.model_name}.pth')

    # Plot and save training history
    plot_training_history(train_losses, val_losses, save_path='hlnet_vae_training_history.png')

if __name__ == '__main__':
    main()
