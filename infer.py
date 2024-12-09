import torch.utils.data as data
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import Model
from sklearn.metrics import auc, precision_recall_curve
from utils import *
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("modality", type= str, help="The path to the input file.") 
arg = parser.parse_args()

class Args:
    def __init__(self, modality):
        # self.modality = 'AUDIO'
        self.modality = modality

        # Test paths
        self.test_rgb_list = '/export/fs05/bodoom1/dl_proj/final_dl/list_exports/rgb_test.list'
        self.test_flow_list = '/export/fs05/bodoom1/dl_proj/final_dl/list_exports/flow_test.list'
        self.test_audio_list = '/export/fs05/bodoom1/dl_proj/final_dl/list_exports/audio_test.list'

        self.gt = '/export/fs05/bodoom1/dl_proj/final_dl/list_exports/gt.npy'
        self.gpus = 1
        self.lr = 0.0001
        self.batch_size = 128
        self.workers = 1  # Reduced from 4 to avoid memory issues
        self.model_name = 'wsanodet'
        self.pretrained_ckpt = None
        if self.modality == 'AUDIO':
            self.feature_size = 128  # 128
        elif self.modality == 'RGB':
            self.feature_size = 1024  # 1024 
        elif self.modality == 'MIX2':
            self.feature_size = 1152  # 1024 + 128

        self.num_classes = 1
        self.dataset_name = 'XD-Violence'
        self.max_seqlen = 200
        self.max_epoch = 50

args = Args(arg.modality)

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
            print(features1.shape, features2.shape)
            if features1.shape[0] == features2.shape[0]:
                features = np.concatenate((features1, features2),axis=1)
            else:
                features = np.concatenate((features1[:-1], features2), axis=1)
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


def create_single_modality_data_loaders(args):
    """
    Create train, validation and test data loaders for a single modality
    """
    print(f"Creating {args.modality} data loaders...")

    # Create new args with only needed attributes

    # List files needed for train/val/test splits
    if args.modality == 'AUDIO':
        args.test_audio_list = args.test_audio_list
    elif args.modality == 'RGB':
        args.test_rgb_list = args.test_rgb_list
    elif args.modality == 'FLOW':
        args.test_flow_list = args.test_flow_list

    test_dataset = Dataset(args, mode='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    print(f"Test loader created with {len(test_dataset)} samples")

    return test_loader


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
    test_loader = create_single_modality_data_loaders(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(args)
    model = model.cuda()

    checkpoint = torch.load(f'/export/fs05/bodoom1/dl_proj/ckpt_{arg.modality.lower()}/wsanodet48.pth', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)


    pr_auc, pr_auc_online = test(test_loader, model, device)
    print('offline pr_auc:{0:.4}; online pr_auc:{1:.4}\n'.format(pr_auc, pr_auc_online))
   