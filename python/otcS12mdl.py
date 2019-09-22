import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import collections
import glob
import numpy as np
import tqdm
import os
import time
import datetime
from pytz import timezone

from otcS12dset import FNMDatasetS12, paddingCollator, load_data
from commonmodules import LinearBlock, LSTMEmbeddingEncoder, lastDLQ
from training_engine import TrainingContext

class OTCS12Module(nn.Module):
    r"""
     Fill up
    """
    def __init__(self, dlq_dim, module_config):
        """
        Parameters:
            seq_n_features: number of features inside sequence. note that first
            9 (0-6 + 12) represent one hot encoding for DLQ
            lstm_size: the size of hidden unit in LSTM 
            linear_size: the number of of output units for the first linear
            embed_dims: the list of tuples, where the first element of tuple
                represents the number of categories and second element of the
                tuple if the embedding size
            embed_drp: drop out percentage after embedding layer
        """
        super(OTCS12Module, self).__init__()
        self.dlq_dim = dlq_dim
        self.model_config = module_config
        self.encoder = LSTMEmbeddingEncoder(
            seq_n_features = module_config['seq_n_features'],
            lstm_conf = module_config['encoder']['lstm'],
            emb_acq_dims = arangeACQEmbeddings(
                module_config['encoder']['emb_acq_dims']
            ),
            emb_seq_dims = arangeSEQEmbeddings(
                module_config['encoder']['emb_seq_dims']
            )
        )
        self.linear = LinearBlock(
            input_size = self.dlq_dim + module_config['encoder']['lstm']['lstm_size'],
            linear_conf = module_config['pipe']
        )
        
        self.output = nn.Linear(self.linear.out_features(), self.dlq_dim * 12)

    def forward(self, seq, seq_len, ymd, acq):
        out = self.encoder(seq, seq_len, ymd, acq)
        dlq = lastDLQ(seq, seq_len, self.dlq_dim)
        out = torch.cat([dlq, out], 1)
        out = self.linear(out)
        out = self.output(out)
        out = out.reshape(-1, self.dlq_dim, 12)
        return out

def arangeACQEmbeddings(emb_acq_dims):
    emb_acq_dict = collections.OrderedDict()
    emb_acq_dict['state_id']            = emb_acq_dims['state_id']
    emb_acq_dict['purpose_id']          = emb_acq_dims['purpose_id']
    emb_acq_dict['mi_type_id']          = emb_acq_dims['mi_type_id']
    emb_acq_dict['occupancy_status_id'] = emb_acq_dims['occupancy_status_id']
    emb_acq_dict['product_type_id']     = emb_acq_dims['product_type_id']
    emb_acq_dict['property_type_id']    = emb_acq_dims['property_type_id']
    emb_acq_dict['seller_id']           = emb_acq_dims['seller_id']
    emb_acq_dict['zip3_id']             = emb_acq_dims['zip3_id']
    return emb_acq_dict

def arangeSEQEmbeddings(emb_seq_dims):
    emb_seq_dict = collections.OrderedDict()
    emb_seq_dict['yyyymm']      = emb_seq_dims['yyyymm']
    emb_seq_dict['msa_id']      = emb_seq_dims['msa_id']
    emb_seq_dict['servicer_id'] = emb_seq_dims['servicer_id']
    return emb_seq_dict

class OTCS12Model(TrainingContext):
    def __init__(self, module_config):
        super(OTCS12Model, self).__init__()
        self.module_config = module_config['module_config']
        self.model = OTCS12Module(19, self.module_config)
        self.optimizer = self.selectOptimizer(self.module_config)
        self.loss_function = nn.CrossEntropyLoss(reduction='sum')

    def selectOptimizer(self, module_config):
        lr = module_config['adam']['lr']
        return optim.Adam(self.model.parameters(), lr=lr)

    def applyModel(self, batch):
        (seq, seq_len, ymd, acq, target) = batch
        seq = seq.to(self.device)
        seq_len = seq_len.to(self.device)
        ymd = ymd.to(self.device)
        acq = acq.to(self.device)
        target_hat = self.model(seq, seq_len, ymd, acq)
        target = target.to(target_hat.device)
        loss = self.loss_function(target_hat, target)
        return loss

    def dataLoaderTrain(self, ds, NUM_WORKERS=6):
        self.trainDL = DataLoader(
            ds, 
            batch_size=self.module_config['batch_size'], 
            shuffle = True,
            collate_fn=paddingCollator, 
            num_workers=NUM_WORKERS, 
            pin_memory=True
        )

    def dataLoaderValid(self, ds, NUM_WORKERS=6):
        self.validDL = DataLoader(
            ds, 
            batch_size=self.module_config['batch_size'], 
            shuffle = True,
            collate_fn=paddingCollator, 
            num_workers=NUM_WORKERS, 
            pin_memory=True
        )

    def loadModel(self, model_path):
        if os.path.exists(model_path):
            print('Loading model checkpoint: {}'.format(model_path))
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint['train_losses']
            self.valid_losses = checkpoint['valid_losses']
            self.checkpoint_epoch = checkpoint['epoch']
        else:
            self.checkpoint_epoch = 0
 
    def saveModel(self, model_path, epoch):
        if type(model.model) == nn.DataParallel:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        torch.save({ 
            'epoch': epoch,
            'model_state_dict': model_state_dict, 
            'model_config' : self.module_config,
            'optimizer_state_dict': self.optimizer.state_dict(), 
            'train_losses' : self.train_losses, 
            'valid_losses' : self.valid_losses 
            }, model_path 
        ) 

def loadDataset(PATH, dlq_dim = 19, oneChunkOnly=True, ratio=0):
    acq, idx_to_seq, seq, _, ym2idx = load_data(
        PATH, 
        verbose=True, 
        oneChunkOnly=oneChunkOnly)
    return FNMDatasetS12 (acq, idx_to_seq, seq, ym2idx, dlq_dim, ratio)


if __name__ == "__main__":
    module_config = {
        'module_config' : {
            'seq_n_features' : 25,
            'batch_size' : 512,
            'adam' : {
                'lr' : 0.010060695099805887
            },
            'encoder': {
                'emb_acq_dims': {
                    'state_id': (55, 11),
                    'purpose_id': (5, 2),
                    'mi_type_id': (4, 2),
                    'occupancy_status_id': (4, 2), 
                    'product_type_id': (2, 2), 
                    'property_type_id': (6, 2), 
                    'seller_id': (95, 24), 
                    'zip3_id': (1001, 45)
                },
                'emb_seq_dims': {
                    'yyyymm' : (219, 11),
                    'msa_id' : (407, 42),
                    'servicer_id' : (46, 6)
                },
                'lstm' : {
                    'lstm_size': 365,
                    'lstm_layers': 4,
                    'lstm_dropout': 0.8568056255089875
                }
            },
            'pipe' : {
                'l1' : (390, 0.1850432102477586),
                'l2' : (74, 0.11613060292987498)
            }
        }
    }

    model = OTCS12Model(module_config)

    TRAIN_PATH = '/home/user/notebooks/data/train'
    VALID_PATH = '/home/user/notebooks/data/valid'
    train_ds = loadDataset(TRAIN_PATH, dlq_dim=19, oneChunkOnly=True, ratio=5)
    valid_ds = loadDataset(VALID_PATH, dlq_dim=19, oneChunkOnly=True, ratio=5)

    model.dataLoaderTrain(train_ds, NUM_WORKERS=6)
    model.dataLoaderValid(valid_ds, NUM_WORKERS=6)
    model.useGPU(True)
    model.makeParallel(True)
    valid_error = model.fit(NUM_EPOCHS=20, save_model=False)

    print(valid_error)

#/***
#/
#100%|█████████████████████████████████████████████████████████████████████████████████| 100/100 [1:57:30<00:00, 64.45s/it, best loss: 3.115884901417626]
#
#{
#    'l1_d': 0.2988637985884944, 
#    'l1_l': 124, 
#    'l2_d': 0.4272170242965795, 
#    'l2_l': 10, 
#    'lr': 0.022183313049507714, 
#    'lstm_dropout': 0.7553171161497654, 
#    'lstm_layers': 2, 
#    'lstm_size': 134, 
#    'msa_id': 7, 
#    'seller_id': 6, 
#    'servicer_id': 1, 
#    'state_id': 11, 
#    'yyyymm': 5, 
#    'zip3_id': 44
#}
#