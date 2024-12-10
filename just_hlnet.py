import torch.utils.data as data
import torch.optim as optim
import numpy as np
import os
import torch
from torch import nn
from datetime import datetime
from model import Model
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import auc, precision_recall_curve

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state_dict = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state_dict = model.state_dict()
            self.counter = 0

    def get_best_model_state(self):
        return self.best_state_dict

class Args:
    def __init__(self):
        self.modality = 'MIX2'
        self.rgb_list = './final_dl/list/rgb.list'
        self.audio_list = './final_dl/list/audio.list'

        # Train paths
        self.train_rgb_list = './final_dl/list/rgb_train.list'
        self.train_audio_list = './final_dl/list/audio_train.list'

        # Val paths
        self.val_rgb_list = './final_dl/list/rgb_val.list'
        self.val_audio_list = './final_dl/list/audio_val.list'

        # Test paths
        self.test_rgb_list = './final_dl/list/rgb_test.list'
        self.test_audio_list = './final_dl/list/audio_test.list'

        self.gt = './final_dl/list/gt.npy'
        self.gpus = 1
        self.lr = 0.0001
        self.batch_size = 128
        self.workers = 1
        self.model_name = 'multimodal_hlnet'
        self.pretrained_ckpt = None
        self.feature_size = 1152  # 1024 + 128 (RGB + Audio)
        self.num_classes = 1
        self.max_seqlen = 200
        self.max_epoch = 50

def process_feat(feat, length, is_random=True):
    if len(feat) > length:
        if is_random:
            return random_extract(feat, length)
        else:
            return uniform_extract(feat, length)
    else:
        return pad(feat, length)

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

class Dataset(data.Dataset):
    def __init__(self, args, transform=None, mode='train'):
        self.modality = args.modality
        self.max_seqlen = args.max_seqlen
        self.transform = transform
        self.test_mode = (mode == 'test')

        # Set appropriate file lists based on mode
        if mode == 'test':
            self.rgb_list_file = args.test_rgb_list
            self.audio_list_file = args.test_audio_list
        elif mode == 'val':
            self.rgb_list_file = args.val_rgb_list
            self.audio_list_file = args.val_audio_list
        else:  # train
            self.rgb_list_file = args.train_rgb_list
            self.audio_list_file = args.train_audio_list

        self._parse_list()

    def _parse_list(self):
        self.list = [line.strip() for line in open(self.rgb_list_file)]
        self.audio_list = [line.strip() for line in open(self.audio_list_file)]

    def __getitem__(self, index):
        # Load RGB features
        file_path1 = self.list[index].strip()
        features1 = np.array(np.load(file_path1), dtype=np.float32)
        label = 0.0 if '_label_A' in file_path1 else 1.0

        # Load corresponding audio features
        audio_index = index // 5  # Since we have 5 RGB frames per audio segment
        file_path2 = self.audio_list[audio_index].strip()
        features2 = np.array(np.load(file_path2), dtype=np.float32)

        # Concatenate features
        features = np.concatenate((features1, features2), axis=1)

        if self.transform is not None:
            features = self.transform(features)

        features = process_feat(features, self.max_seqlen, is_random=not self.test_mode)
        return features, label

    def __len__(self):
        return len(self.list)

def CLAS(logits, label, seq_len, criterion, device, is_topk=True):
    logits = logits.squeeze()
    instance_logits = torch.zeros(0).to(device)
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
    instance_logits = torch.tensor(0).to(device)
    for i in range(logits.shape[0]):
        tmp1 = torch.sigmoid(logits[i, :seq_len[i]]).squeeze()
        tmp2 = torch.sigmoid(logits2[i, :seq_len[i]]).squeeze()
        loss = torch.mean(-tmp1.detach() * torch.log(tmp2))
        instance_logits = instance_logits + loss
    instance_logits = instance_logits/logits.shape[0]
    return instance_logits

def validate_epoch(val_loader, model, criterion, device, is_topk):
    """Run validation for one epoch"""
    model.eval()
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for input, label in val_loader:
            seq_len = torch.sum(torch.max(torch.abs(input), dim=2)[0]>0, 1)
            input = input[:, :torch.max(seq_len), :]
            input, label = input.float().to(device), label.float().to(device)

            logits, logits2 = model(input, seq_len)
            clsloss = CLAS(logits, label, seq_len, criterion, device, is_topk)
            clsloss2 = CLAS(logits2, label, seq_len, criterion, device, is_topk)
            croloss = CENTROPY(logits, logits2, seq_len, device)

            total_loss += (clsloss + clsloss2 + 5*croloss).item()
            batch_count += 1

    return total_loss / batch_count

def test_hlnet(dataloader, model, device):
    """Test function that evaluates video-level predictions"""
    print("Starting test...")
    with torch.no_grad():
        model.eval()
        pred = []
        pred2 = []
        gt = []

        for i, (input, label) in enumerate(dataloader):
            gt.append(label[0].item())
            input = input.to(device)
            logits, logits2 = model(inputs=input, seq_len=None)

            # Get predictions for offline model
            logits = torch.squeeze(logits)
            sig = torch.sigmoid(logits)
            batch_pred = torch.mean(sig).item()
            pred.append(batch_pred)

            # Get predictions for online model
            logits2 = torch.squeeze(logits2)
            sig2 = torch.sigmoid(logits2)
            batch_pred2 = torch.mean(sig2).item()
            pred2.append(batch_pred2)

        # Convert to numpy arrays
        gt = np.array(gt)
        pred = np.array(pred)
        pred2 = np.array(pred2)

        # Calculate metrics
        precision, recall, _ = precision_recall_curve(gt, pred)
        pr_auc = auc(recall, precision)

        precision2, recall2, _ = precision_recall_curve(gt, pred2)
        pr_auc2 = auc(recall2, precision2)

        return pr_auc, pr_auc2

def train_hlnet(train_loader, model, optimizer, scheduler, criterion,
                device, is_topk, val_loader=None):
    """Training function with loss tracking"""
    model.train()
    epoch_loss = 0.0
    batch_count = 0

    for i, (input, label) in enumerate(train_loader):
        seq_len = torch.sum(torch.max(torch.abs(input), dim=2)[0]>0, 1)
        input = input[:, :torch.max(seq_len), :]
        input, label = input.float().to(device), label.float().to(device)

        logits, logits2 = model(input, seq_len)
        clsloss = CLAS(logits, label, seq_len, criterion, device, is_topk)
        clsloss2 = CLAS(logits2, label, seq_len, criterion, device, is_topk)
        croloss = CENTROPY(logits, logits2, seq_len, device)

        total_loss = clsloss + clsloss2 + 5*croloss

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
        val_epoch_loss = validate_epoch(val_loader, model, criterion, device, is_topk)

    return model, avg_epoch_loss, val_epoch_loss

def create_data_loaders(args):
    """Create train, validation and test data loaders"""
    print("Creating data loaders...")

    train_dataset = Dataset(args, mode='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    print(f"Train loader created with {len(train_dataset)} samples")

    val_dataset = Dataset(args, mode='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    print(f"Validation loader created with {len(val_dataset)} samples")

    test_dataset = Dataset(args, mode='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=5,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    print(f"Test loader created with {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader

def plot_training_history(train_losses, val_losses, save_path='hlnet_training_history.png'):
    """Plot training and validation losses"""
    plt.figure(figsize=(12, 8))
    epochs = range(len(train_losses))

    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')

    plt.title('HL-Net Multimodal Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Initialize settings
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model and move to device
    model = Model(args).to(device)

    # Setup optimizer
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

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(args)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=7, verbose=True)

    # Training tracking
    train_losses = []
    val_losses = []
    best_pr_auc = 0
    best_epoch = 0

    print(f"Starting training on {device}")

    for epoch in range(args.max_epoch):
        print(f"\nEpoch {epoch+1}/{args.max_epoch}")
        st = time.time()

        # Training step
        model, train_loss, val_loss = train_hlnet(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            is_topk=is_topk,
            val_loader=val_loader
        )

        # Store losses
        train_losses.append(train_loss)
        if val_loss is not None:
            val_losses.append(val_loss)

        # Calculate PR-AUC for monitoring
        pr_auc, pr_auc_online = test_hlnet(test_loader, model, device)

        # Update best PR-AUC
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_epoch = epoch
            # Save best model
            save_dir = './only_hlnet_saves_mm/best'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{save_dir}/{args.model_name}_best.pth')

        print(f'Epoch {epoch+1}/{args.max_epoch}:')
        print(f'Train Loss: {train_loss:.4f}')
        if val_loss is not None:
            print(f'Validation Loss: {val_loss:.4f}')
        print(f'PR-AUC (Offline/Online): {pr_auc:.4f}/{pr_auc_online:.4f}')
        print(f'Best PR-AUC: {best_pr_auc:.4f} (Epoch {best_epoch+1})')
        print(f'Epoch time: {time.time() - st:.2f}s')

        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            model.load_state_dict(early_stopping.get_best_model_state())
            break

        scheduler.step()

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0 and epoch > 0:
            save_dir = './only_hlnet_saves_mm/checkpoints'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{save_dir}/{args.model_name}_epoch{epoch}.pth')

    # Save final model
    save_dir = './only_hlnet_saves_mm/final'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{save_dir}/{args.model_name}_final.pth')

    # Plot and save training history
    plot_training_history(train_losses, val_losses, save_path='only_hlnet_training_history.png')

    # Final evaluation
    final_pr_auc, final_pr_auc_online = test_hlnet(test_loader, model, device)
    print("\nTraining completed!")
    print(f'Final PR-AUC (Offline/Online): {final_pr_auc:.4f}/{final_pr_auc_online:.4f}')
    print(f'Best PR-AUC: {best_pr_auc:.4f} (Epoch {best_epoch+1})')

if __name__ == '__main__':
    main()
