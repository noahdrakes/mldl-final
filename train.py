import torch.utils.data as data
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import Model
from layers import GraphAttentionLayer, linear, GraphConvolution, SimilarityAdj
from sklearn.metrics import auc, precision_recall_curve
from utils import *
import time

class Args:
    def __init__(self):
        self.modality = 'AUDIO'
        # Original paths
        self.rgb_list = '/export/fs05/bodoom1/dl_proj/final_dl/list/rgb.list'
        self.flow_list = '/export/fs05/bodoom1/dl_proj/final_dl/list/flow.list'
        self.audio_list = '/export/fs05/bodoom1/dl_proj/final_dl/list/audio.list'

        # Train paths
        self.train_rgb_list = '/export/fs05/bodoom1/dl_proj/final_dl/list/rgb_train.list'
        self.train_flow_list = '/export/fs05/bodoom1/dl_proj/final_dl/list/flow_train.list'
        self.train_audio_list = '/export/fs05/bodoom1/dl_proj/final_dl/list/audio_train.list'

        # Val paths
        self.val_rgb_list = '/export/fs05/bodoom1/dl_proj/final_dl/list/rgb_val.list'
        self.val_flow_list = '/export/fs05/bodoom1/dl_proj/final_dl/list/flow_val.list'
        self.val_audio_list = '/export/fs05/bodoom1/dl_proj/final_dl/list/audio_val.list'

        # Test paths
        self.test_rgb_list = '/export/fs05/bodoom1/dl_proj/final_dl/list/rgb_test.list'
        self.test_flow_list = '/export/fs05/bodoom1/dl_proj/final_dl/list/flow_test.list'
        self.test_audio_list = '/export/fs05/bodoom1/dl_proj/final_dl/list/audio_test.list'

        self.gt = '/export/fs05/bodoom1/dl_proj/final_dl/list/gt.npy'
        self.gpus = 1
        self.lr = 0.0001
        self.batch_size = 128
        self.workers = 1  # Reduced from 4 to avoid memory issues
        self.model_name = 'wsanodet'
        self.pretrained_ckpt = None
        #self.feature_size = 1152  # 1024 + 128
        self.feature_size = 128  # 1024 + 128
        self.num_classes = 1
        self.dataset_name = 'XD-Violence'
        self.max_seqlen = 200
        self.max_epoch = 50

args = Args()
    



# from utils import process_feat

class Dataset(data.Dataset):
    def __init__(self, args, transform=None, mode='train'):
        self.modality = args.modality
        """
        Args:
            args: Arguments containing dataset paths and configuration
            transform: Optional transforms to apply
            mode: One of ['train', 'val', 'test'] to specify the dataset split
        """

        if mode == 'test':
            self.rgb_list_file = args.test_rgb_list
            self.flow_list_file = args.test_flow_list
            self.audio_list_file = args.test_audio_list
        elif mode == 'val':
            self.rgb_list_file = args.val_rgb_list
            self.flow_list_file = args.val_flow_list
            self.audio_list_file = args.val_audio_list
        else: # train
            self.rgb_list_file = args.train_rgb_list
            self.flow_list_file = args.train_flow_list
            self.audio_list_file = args.train_audio_list

        self.max_seqlen = args.max_seqlen
        self.tranform = transform
        self.test_mode = (mode == 'test')
        self.normal_flag = '_label_A'
        self._parse_list()

    def _parse_list(self):
        if self.modality == 'AUDIO':
            self.list = list(open(self.audio_list_file))
        elif self.modality == 'RGB':
            self.list = list(open(self.rgb_list_file))
            print("here")
            # print(self.list)
        elif self.modality == 'FLOW':
            self.list = list(open(self.flow_list_file))
        elif self.modality == 'MIX':
            self.list = list(open(self.rgb_list_file))
            self.flow_list = list(open(self.flow_list_file))
        elif self.modality == 'MIX2':
            self.list = list(open(self.rgb_list_file))
            self.audio_list = list(open(self.audio_list_file))
        elif self.modality == 'MIX3':
            self.list = list(open(self.flow_list_file))
            self.audio_list = list(open(self.audio_list_file))
        elif self.modality == 'MIX_ALL':
            self.list = list(open(self.rgb_list_file))
            self.flow_list = list(open(self.flow_list_file))
            self.audio_list = list(open(self.audio_list_file))
        else:
            assert 1 > 2, 'Modality is wrong!'

    def __getitem__(self, index):
        if self.normal_flag in self.list[index]:
            label = 0.0
        else:
            label = 1.0

        if self.modality == 'AUDIO':
            features = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
        elif self.modality == 'RGB':
            features = np.array(np.load(self.list[index].strip('\n')),dtype=np.float32)
        elif self.modality == 'FLOW':
            features = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
        elif self.modality == 'MIX':
            features1 = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
            features2 = np.array(np.load(self.flow_list[index].strip('\n')), dtype=np.float32)
            if features1.shape[0] == features2.shape[0]:
                features = np.concatenate((features1, features2),axis=1)
            else:# because the frames of flow is one less than that of rgb
                features = np.concatenate((features1[:-1], features2), axis=1)
        elif self.modality == 'MIX2':
            features1 = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
            features2 = np.array(np.load(self.audio_list[index//5].strip('\n')), dtype=np.float32)
        elif self.modality == 'MIX3':
            features1 = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
            features2 = np.array(np.load(self.audio_list[index//5].strip('\n')), dtype=np.float32)
            if features1.shape[0] == features2.shape[0]:
                features = np.concatenate((features1, features2),axis=1)
            else:# because the frames of flow is one less than that of rgb
                features = np.concatenate((features1[:-1], features2), axis=1)
        elif self.modality == 'MIX_ALL':
            features1 = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
            features2 = np.array(np.load(self.flow_list[index].strip('\n')), dtype=np.float32)
            features3 = np.array(np.load(self.audio_list[index//5].strip('\n')), dtype=np.float32)
            if features1.shape[0] == features2.shape[0]:
                features = np.concatenate((features1, features2, features3),axis=1)
            else:# because the frames of flow is one less than that of rgb
                features = np.concatenate((features1[:-1], features2, features3[:-1]), axis=1)
        else:
            assert 1>2, 'Modality is wrong!'
        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features, label

        else: 
            features = process_feat(features, self.max_seqlen, is_random=False)
            return features, label

    def __len__(self):
        return len(self.list)




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
        batch_size=1,  # Using smaller batch size for testing
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    print(f"Test loader created with {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = create_data_loaders(args)

def create_single_modality_data_loaders(args, modality='AUDIO'):
    """
    Create train, validation and test data loaders for a single modality
    """
    print(f"Creating {modality} data loaders...")

    # Create new args with only needed attributes
    args_new = Args()
    args_new.modality = modality

    # List files needed for train/val/test splits
    if modality == 'AUDIO':
        args_new.train_audio_list = args.train_audio_list
        args_new.val_audio_list = args.val_audio_list
        args_new.test_audio_list = args.test_audio_list
    elif modality == 'RGB':
        args_new.train_rgb_list = args.train_rgb_list
        args_new.val_rgb_list = args.val_rgb_list
        args_new.test_rgb_list = args.test_rgb_list
    elif modality == 'FLOW':
        args_new.train_flow_list = args.train_flow_list
        args_new.val_flow_list = args.val_flow_list
        args_new.test_flow_list = args.test_flow_list

    # Create data loaders
    train_dataset = Dataset(args_new, mode='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=args_new.batch_size,
        shuffle=True,
        num_workers=args_new.workers,
        pin_memory=True
    )
    print(f"Train loader created with {len(train_dataset)} samples")

    val_dataset = Dataset(args_new, mode='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=args_new.batch_size,
        shuffle=False,
        num_workers=args_new.workers,
        pin_memory=True
    )
    print(f"Validation loader created with {len(val_dataset)} samples")

    test_dataset = Dataset(args_new, mode='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args_new.workers,
        pin_memory=True
    )
    print(f"Test loader created with {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader


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


def validate(dataloader, model, criterion, device, is_topk):
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        count = 0
        for i, (input, label) in enumerate(dataloader):
            seq_len = torch.sum(torch.max(torch.abs(input), dim=2)[0]>0, 1)
            input = input[:, :torch.max(seq_len), :]
            input, label = input.float().to(device), label.float().to(device)
            logits, logits2 = model(input, seq_len)
            clsloss = CLAS(logits, label, seq_len, criterion, device, is_topk)
            clsloss2 = CLAS(logits2, label, seq_len, criterion, device, is_topk)
            croloss = CENTROPY(logits, logits2, seq_len, device)

            batch_loss = clsloss + clsloss2 + 5*croloss
            total_loss += batch_loss.item()
            count += 1
        return total_loss / count if count > 0 else 0.0


def train(dataloader, model, optimizer, scheduler, criterion, device, is_topk, val_loader=None):
    model.train()
    running_loss = 0.0
    count = 0
    for i, (input, label) in enumerate(dataloader):
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
        scheduler.step()

        running_loss += total_loss.item()
        count += 1

        # Print training loss every 100 steps
        if (i + 1) % 100 == 0:
            avg_train_loss = running_loss / count
            print(f"Step {i+1}: Average Training Loss: {avg_train_loss:.4f}")
            running_loss = 0.0
            count = 0

            # If val_loader is provided, evaluate on validation set
            if val_loader is not None:
                val_loss = validate(val_loader, model, criterion, device, is_topk)
                print(f"Step {i+1}: Validation Loss: {val_loss:.4f}")

    return model


def test(dataloader, model, device):
    gt =[]
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(device)
        pred2 = torch.zeros(0).to(device)
        for i, (input, label) in enumerate(dataloader):
            gt.append(label)
            input = input.to(device)
            logits, logits2 = model(inputs=input, seq_len=None)
            logits = torch.squeeze(logits)
            sig = torch.sigmoid(logits)
            sig = torch.mean(sig, 0)
            sig = sig.unsqueeze(0)
            pred = torch.cat((pred, sig))
            '''
            online detection
            '''
            logits2 = torch.squeeze(logits2)
            sig2 = torch.sigmoid(logits2)
            sig2 = torch.mean(sig2, 0)
            sig2 = sig2.unsqueeze(0)
            pred2 = torch.cat((pred2, sig2))

        pred = list(pred.cpu().detach().numpy())
        pred2 = list(pred2.cpu().detach().numpy())




        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        precision, recall, th = precision_recall_curve(list(gt), pred2)
        pr_auc2 = auc(recall, precision)
        return pr_auc, pr_auc2
     




if __name__ == '__main__':
    aligned_files = find_matching_files()

    # Create train/val splits
    split_data = create_splits(aligned_files)

        # Write list files
    write_list_files(split_data, aligned_files, "/export/fs05/bodoom1/dl_proj/final_dl/list")

    train_loader, val_loader, test_loader = create_single_modality_data_loaders(args)

    
        

    device = torch.device("cuda")
    # train_loader = DataLoader(Dataset(args, mode='train'),
    #                         batch_size=args.batch_size, shuffle=True,
    #                         num_workers=args.workers, pin_memory=True)
    # test_loader = DataLoader(Dataset(args, mode='test'),
    #                         batch_size=5, shuffle=False,
    #                         num_workers=args.workers, pin_memory=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(args)
    model = model.cuda()

    for name, value in model.named_parameters():
        print(name)
    approximator_param = list(map(id, model.approximator.parameters()))
    approximator_param += list(map(id, model.conv1d_approximator.parameters()))
    base_param = filter(lambda p: id(p) not in approximator_param, model.parameters())

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')
    optimizer = optim.Adam([{'params': base_param},
                            {'params': model.approximator.parameters(), 'lr': args.lr / 2},
                            {'params': model.conv1d_approximator.parameters(), 'lr': args.lr / 2},
                            ],
                            lr=args.lr, weight_decay=0.000)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
    criterion = torch.nn.BCELoss()

    is_topk = True
    gt = np.load(args.gt)
    pr_auc, pr_auc_online = test(test_loader, model, device)
    print('Random initalization: offline pr_auc:{0:.4}; online pr_auc:{1:.4}\n'.format(pr_auc, pr_auc_online))
    for epoch in range(args.max_epoch): 
        st = time.time()
        model = train(train_loader, model, optimizer, scheduler, criterion, device, is_topk)
        if epoch % 2 == 0 and not epoch == 0:
            torch.save(model.state_dict(), '/export/fs05/bodoom1/dl_proj/ckpt/'+args.model_name+'{}.pth'.format(epoch))

        pr_auc, pr_auc_online = test(test_loader, model, device)
        print('Epoch {0}/{1}: offline pr_auc:{2:.4}; online pr_auc:{3:.4}\n'.format(epoch, args.max_epoch, pr_auc, pr_auc_online))
    torch.save(model.state_dict(), '/export/fs05/bodoom1/dl_proj/ckpt/' + args.model_name + '.pth')
