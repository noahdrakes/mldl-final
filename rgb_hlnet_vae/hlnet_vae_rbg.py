import os
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import torch
from torch import nn
from datetime import datetime
from model import Model
from VAE import VAE
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import auc, precision_recall_curve

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0, verbose=False):
        """
        Args:
            patience (int): Number of epochs to wait before stopping if no improvement
            min_delta (float): Minimum change in monitored value to qualify as an improvement
            verbose (bool): If True, prints out early stopping information
        """
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
        self.modality = 'RGB'  # Changed to RGB only
        self.rgb_list = './final_dl/list/rgb.list'

        # Train/Val/Test paths
        self.train_rgb_list = './final_dl/list/rgb_train.list'
        self.val_rgb_list = './final_dl/list/rgb_val.list'
        self.test_rgb_list = './final_dl/list/rgb_test.list'

        self.gt = './final_dl/list/gt.npy'
        self.gpus = 1
        self.lr = 0.0001
        self.batch_size = 128
        self.workers = 1
        self.model_name = 'rgb_wsanodet'
        self.pretrained_ckpt = None
        self.feature_size = 1024  # Changed to RGB feature size
        self.num_classes = 1
        self.max_seqlen = 200
        self.max_epoch = 200

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

        # Set appropriate file list based on mode
        if mode == 'test':
            self.rgb_list_file = args.test_rgb_list
        elif mode == 'val':
            self.rgb_list_file = args.val_rgb_list
        else:  # train
            self.rgb_list_file = args.train_rgb_list

        self._parse_list()

    def _parse_list(self):
        self.list = [line.strip() for line in open(self.rgb_list_file)]

    def __getitem__(self, index):
        file_path = self.list[index].strip()
        features = np.array(np.load(file_path), dtype=np.float32)
        label = 0.0 if '_label_A' in file_path else 1.0

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

def validate_epoch(val_loader, hlnet, vae, criterion, device, is_topk,
                  HLNET_LOSS_WEIGHT, RECON_LOSS_WEIGHT):
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
    print("Starting test...")
    with torch.no_grad():
        model.eval()
        pred = []
        pred2 = []
        gt = []

        for i, (input, label) in enumerate(dataloader):
            if i == 0:  # Print shapes for first batch
                print(f"Input shape: {input.shape}")
                print(f"Label shape: {label.shape}")

            # For ground truth, just take the label of first frame in batch
            # (they should all be same for a video segment)
            gt.append(label[0].item())

            input = input.to(device)
            logits, logits2 = model(inputs=input, seq_len=None)
            if i == 0:
                print(f"Logits shape: {logits.shape}")

            # Get one prediction per batch/video segment
            logits = torch.squeeze(logits)
            sig = torch.sigmoid(logits)
            if i == 0:
                print(f"Sig shape before mean: {sig.shape}")

            # Average over both frames and sequence length to get one score per video
            batch_pred = torch.mean(sig).item()
            pred.append(batch_pred)

            # Same for online predictions
            logits2 = torch.squeeze(logits2)
            sig2 = torch.sigmoid(logits2)
            batch_pred2 = torch.mean(sig2).item()
            pred2.append(batch_pred2)

        # Convert to numpy arrays
        gt = np.array(gt)
        pred = np.array(pred)
        pred2 = np.array(pred2)

        print(f"Final shapes - GT: {gt.shape}, Pred: {pred.shape}, Pred2: {pred2.shape}")

        precision, recall, th = precision_recall_curve(gt, pred)
        pr_auc = auc(recall, precision)
        precision, recall, th = precision_recall_curve(gt, pred2)
        pr_auc2 = auc(recall, precision)
        return pr_auc, pr_auc2

def train_hlnet_vae(train_loader, hlnet, vae, optimizer, scheduler, criterion,
                    device, is_topk, HLNET_LOSS_WEIGHT, RECON_LOSS_WEIGHT, val_loader=None):
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

def plot_training_history(train_losses, val_losses, save_path=''):
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
    # Initialize VAE
    args_vae = Args()
    args_vae.feature_size = 1024  # RGB feature size
    args_vae.batch_size = 64
    args_vae.modality = 'RGB'
    args_vae.max_epoch = 200
    args_vae.lr = 0.0005

    vae_model = VAE(latent_dim=64, input_dim=args_vae.feature_size, seq_len=200)
    vae_dir = "./single_modality/vae_checkpoints/best_trained_vae_RGB.pkl"
    vae_model.load_state_dict(torch.load(vae_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_model = vae_model.to(device)
    vae_model.eval()

    HL_NET_LOSS_weight = 0.8
    RECON_LOSS_weight = 0.2

    # Initialize HL-Net
    args = Args()
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

    # Create datasets and dataloaders
    train_loader, val_loader, test_loader = create_data_loaders(args)

    # Track losses
    train_losses = []
    val_losses = []
    best_pr_auc = 0
    best_epoch = 0

    early_stopping = EarlyStopping(patience=7, verbose=True)
    for epoch in range(args.max_epoch):
        print(f"\nEpoch {epoch+1}/{args.max_epoch}")
        st = time.time()

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
        train_losses.append(train_loss)

        if val_loss is not None:
            val_losses.append(val_loss)

        pr_auc, pr_auc_online = test_hl_vae(test_loader, model, device)

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_epoch = epoch

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
            # Load the best model state
            model.load_state_dict(early_stopping.get_best_model_state())
            break

        scheduler.step()

        if epoch % 5 == 0 and epoch > 0:
            save_dir = './hlnet_saves_rgb'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{save_dir}/{args.model_name}{epoch}.pth')

    save_dir = './hlnet_saves_rgb/final'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{save_dir}/rgb_{args.model_name}{epoch}.pth')
    plot_training_history(train_losses, val_losses, save_path='hlnet_vae_rgb_training_history.png')

if __name__ == '__main__':
    main()
