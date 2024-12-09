import torch.utils.data as data
import numpy as np
import os
import glob
import random
from pathlib import Path
import torch
from torch import nn
import torch
from datetime import datetime

class Args:
    def __init__(self):
        self.modality = 'MIX2'
        # Original paths
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


class Sampling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_means, z_log_vars):
        epsilon = torch.randn_like(z_means, dtype=torch.float32)
        return z_means + torch.exp(0.5 * z_log_vars) * epsilon

class Encoder(nn.Module):
    def __init__(self, latent_dim, input_dim=1152, seq_len=200):
        super().__init__()
        self.latent_dim = latent_dim

        # Reduced number of feature maps in encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, stride=2, padding=1),  # Reduced from 576
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 128, kernel_size=3, stride=2, padding=1),  # Reduced from 288
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1),  # Reduced from 144
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Flatten()
        )

        flattened_dim = 64 * 25  # Updated based on reduced features

        self.lin_mean = nn.Sequential(
            nn.Linear(flattened_dim, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )

        self.lin_log_var = nn.Sequential(
            nn.Linear(flattened_dim, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
#
        self.sampling = Sampling()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        z_means = self.lin_mean(x)
        z_log_vars = self.lin_log_var(x)
        z = self.sampling(z_means, z_log_vars)
        return z, z_means, z_log_vars

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim=1152, seq_len=200):
        super().__init__()
        self.seq_len = seq_len
        flattened_dim = 64 * 25  # Updated based on reduced features

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, flattened_dim),
            nn.BatchNorm1d(flattened_dim),
            nn.ReLU(True)
        )

        # Reduced number of feature maps in decoder
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Reduced from 144->288
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # Reduced from 288->576
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.ConvTranspose1d(256, input_dim, kernel_size=3, stride=2, padding=1, output_padding=1),  # Reduced from 576->input_dim
            nn.BatchNorm1d(input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder_fc(x)
        x = x.view(-1, 64, 25)  # Updated based on reduced features
        x = self.decoder_conv(x)
        x = x.permute(0, 2, 1)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim, input_dim=1152, seq_len=200):
        super().__init__()
        self.encoder = Encoder(latent_dim, input_dim, seq_len)
        self.decoder = Decoder(latent_dim, input_dim, seq_len)

    def forward(self, x):
        z, z_means, z_log_vars = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z_means, z_log_vars


class NormalDataset(data.Dataset):
    def __init__(self, args, transform=None, mode='train'):
        self.modality = args.modality
        self.normal_flag = '_label_A'
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
        """Parse file lists and filter for normal samples only"""
        def filter_normal_samples(file_list):
            return [f for f in file_list if self.normal_flag in f]

        if self.modality == 'AUDIO':
            self.list = filter_normal_samples(list(open(self.audio_list_file)))
        elif self.modality == 'RGB':
            self.list = filter_normal_samples(list(open(self.rgb_list_file)))
        elif self.modality == 'FLOW':
            self.list = filter_normal_samples(list(open(self.flow_list_file)))
        elif self.modality == 'MIX2':
            # For MIX2, we need to handle the 5:1 ratio between RGB and audio
            self.list = filter_normal_samples(list(open(self.rgb_list_file)))
            # Filter audio list and ensure alignment
            all_audio = list(open(self.audio_list_file))
            self.audio_list = [f for f in all_audio if self.normal_flag in f]

            # Ensure RGB and audio lists are aligned (5:1 ratio)
            rgb_video_ids = set([self._get_video_id(f) for f in self.list])
            audio_video_ids = set([self._get_video_id(f) for f in self.audio_list])
            common_ids = rgb_video_ids & audio_video_ids

            # Filter lists to only include common videos
            self.list = [f for f in self.list if self._get_video_id(f) in common_ids]
            self.audio_list = [f for f in self.audio_list if self._get_video_id(f) in common_ids]

    def _get_video_id(self, filepath):
        """Extract video ID from filepath"""
        filename = os.path.basename(filepath.strip('\n'))
        return filename.split('_label')[0]

    def __getitem__(self, index):
        if self.modality in ['RGB', 'FLOW', 'AUDIO']:
            # Remove '/content/' prefix from the path
            file_path = self.list[index].strip('\n').replace('/content/', '')
            features = np.array(np.load(file_path), dtype=np.float32)
        elif self.modality == 'MIX2':
            # Load RGB features
            file_path1 = self.list[index].strip('\n').replace('/content/', '')
            features1 = np.array(np.load(file_path1), dtype=np.float32)
            # Load corresponding audio features
            audio_index = index // 5
            file_path2 = self.audio_list[audio_index].strip('\n').replace('/content/', '')
            features2 = np.array(np.load(file_path2), dtype=np.float32)
    
            # Handle potential dimension mismatch
            if features1.shape[0] > features2.shape[0]:
                features1 = features1[:features2.shape[0]]
            features = np.concatenate((features1, features2), axis=1)
    
        if self.transform is not None:
            features = self.transform(features)
    
        features = process_feat(features, self.max_seqlen, is_random=not self.test_mode)
        return features, 0.0

    def __len__(self):
        return len(self.list)

def create_normal_data_loaders(args):
    """Create data loaders for normal samples only"""
    print("Creating normal-only data loaders...")

    # Create train loader
    train_dataset = NormalDataset(args, mode='train')
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    print(f"Normal train loader created with {len(train_dataset)} samples")

    # Create validation loader
    val_dataset = NormalDataset(args, mode='val')
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    print(f"Normal validation loader created with {len(val_dataset)} samples")

    # Create test loader
    test_dataset = NormalDataset(args, mode='test')
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    print(f"Normal test loader created with {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader

def process_feat(feat, length, is_random=True):
    """Process features to have consistent length"""
    if len(feat) > length:
        if is_random:
            r = np.random.randint(len(feat) - length)
            return feat[r:r + length]
        else:
            r = np.linspace(0, len(feat) - 1, length, dtype=np.uint16)
            return feat[r, :]
    else:
        return np.pad(feat, ((0, length - len(feat)), (0, 0)), mode='constant', constant_values=0)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def validate_vae(vae, val_loader, device):
    """Run validation loop and return average loss"""
    vae.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    n_samples = 0

    with torch.no_grad():
        for data, labels in val_loader:
            # Only process normal samples (label == 0)
            normal_mask = (labels == 0.0)
            if not normal_mask.any():
                continue

            data = data[normal_mask].to(device)
            recon_data, mu, logvar = vae(data)

            # Reconstruction loss
            recon_criterion = torch.nn.MSELoss(reduction='sum')
            recon_loss = recon_criterion(recon_data, data)

            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Total loss
            loss = recon_loss + kl_loss

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            n_samples += data.size(0)

    # Calculate averages
    if n_samples > 0:
        avg_loss = total_loss / n_samples
        avg_recon = total_recon_loss / n_samples
        avg_kl = total_kl_loss / n_samples
    else:
        avg_loss = float('inf')
        avg_recon = float('inf')
        avg_kl = float('inf')

    vae.train()
    return avg_loss, avg_recon, avg_kl

def train_vae(vae, train_loader, val_loader, args, save_dir='vae_checkpoints'):
    """Main training loop for VAE"""

    # Create directory for saving checkpoints
    os.makedirs(save_dir, exist_ok=True)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=5)
    history = {
        'train_loss': [],
        'train_recon': [],
        'train_kl': [],
        'val_loss': [],
        'val_recon': [],
        'val_kl': []
    }
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.max_epoch):
        # Training
        vae.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        n_samples = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            # Only process normal samples (label == 0)
            normal_mask = (labels == 0.0)
            if not normal_mask.any():
                continue

            data = data[normal_mask].to(device)
            optimizer.zero_grad()

            # Forward pass
            recon_data, mu, logvar = vae(data)

            # Losses
            recon_criterion = torch.nn.MSELoss(reduction='sum')
            recon_loss = recon_criterion(recon_data, data)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Record losses
            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
            n_samples += data.size(0)

        # Calculate average training losses
        if n_samples > 0:
            avg_train_loss = train_loss / n_samples
            avg_train_recon = train_recon / n_samples
            avg_train_kl = train_kl / n_samples
        else:
            print("Warning: No normal samples in training batch")
            continue

        # Validation
        val_loss, val_recon, val_kl = validate_vae(vae, val_loader, device)

        # Print progress
        print(f'Epoch {epoch+1}/{args.max_epoch}:')
        print(f'Training - Loss: {avg_train_loss:.4f}, Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f}')
        print(f'Validation - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}\n')

        history['train_loss'].append(avg_train_loss)
        history['train_recon'].append(avg_train_recon)
        history['train_kl'].append(avg_train_kl)
        history['val_loss'].append(val_loss)
        history['val_recon'].append(val_recon)
        history['val_kl'].append(val_kl)
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(save_dir, f'vae_{args.modality}_best_{timestamp}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'history': history
            }, save_path)
            print(f'Saved best model to {save_path}')
            torch.save(vae.state_dict(), os.path.join(save_dir, f"best_trained_vae_{args.modality}.pkl"))


        # Early stopping
        if early_stopping(val_loss):
            print("Early stopping triggered")
            break

    return vae

def main():
    # adjust args needed for VAE
    args_vae = Args()
    args_vae.feature_size = 128  # 1024 (RGB) + 128 (audio)
    args_vae.batch_size = 64
    args_vae.modality = 'AUDIO'
    args_vae.max_epoch = 500
    args_vae.lr = 0.0005
    
    # initialize VAE with correct input dimension
    vae = VAE(latent_dim=64, input_dim=args_vae.feature_size, seq_len=200)
    
    # Create normal-only dataloaders
    normal_train_loader, normal_val_loader, normal_test_loader = create_normal_data_loaders(args_vae)
    
    # Train the VAE
    trained_vae = train_vae(vae, normal_train_loader, normal_val_loader, args_vae)
    # Save the model
    #torch.save(trained_vae.state_dict(), "./last_trained_vae.pkl")

if __name__ == '__main__':
    main()
