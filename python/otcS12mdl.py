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
            emb_acq_dims = module_config['encoder']['emb_acq_dims'],
            emb_seq_dims = module_config['encoder']['emb_seq_dims']
        )
        self.linear = LinearBlock(
            input_size = self.dlq_dim + module_config['encoder']['lstm']['lstm_size'],
            linear_conf = module_config['pipe']
        )
        

    def forward(self, X):
        (seq, seq_len, ymd, acq) = X
        enc_out = self.encoder(seq, seq_len, ymd, acq)
        dlq = lastDLQ(seq, seq_len, self.dlq_dim)
        lin_in = torch.cat(dlq.unsqueeze(1), enc_out, 1)
        out = self.linear(lin_in)
        return out

class OTCS12Model(TrainingContext):
    def __init__(self, module_config):
        super(OTCS12Model, self).__init__()
        self.module_config = module_config['module_config']
        self.model = OTCS12Module(9, self.module_config)
        self.optiminer = self.selectOptimizer(self.module_config)
        self.loss_function = nn.CrossEntropyLoss(reduction='sum')

    def selectOptimizer(self, module_config):
        lr = module_config['adam']['lr']
        return optim.Adam(self.model.parameters(), lr=lr)

    def applyModel(self, batch):
        (seq, seq_len, ymd, acq, target) = batch
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

    def fit(self, NUM_EPOCHS, save_model=False):
        def lprint(x):
            return "|".join(map(lambda x: "{:.4f}".format(x), x))

        ch = self.checkpoint_epoch
        try:
            with tqdm.trange(ch, ch + NUM_EPOCHS) as t:
                for epoch in t:
                    t.set_description('Epoch: %i' % epoch)
                    train_loss = self.trainLoop(epoch)
                    valid_loss = self.validLoop(epoch)

                    t.set_postfix(
                        TL = "{:.4f}".format(train_loss),
                        MVL = "{:.4f}".format(np.mean(self.valid_losses[-10:])),
                        VL = lprint(self.valid_losses[-4:])
                    )

                    if len(self.valid_losses) > 10 \
                        and valid_loss > np.mean(self.valid_losses[-10:]):
                        print('Validation loss is increasing quit before saving')
                        break

                    if save_model:
                        self.saveModel(epoch)
        except KeyboardInterrupt:
            if save_model:
                print ('Saving the model state before exiting')
                self.saveModel(epoch)

        return self.valid_losses[-1]


if __name__ == "__main__":
    module_config = {
        'module_config' : {
            'seq_n_features' : 32,
            'batch_size' : 512,
            'adam' : {
                        'lr' : 0.01
            },
            'encoder': {
                'emb_acq_dims': {
                    'state_id': (55, 20),
                    'purpose_id': (5, 2),
                    'mi_type_id': (4, 2),
                    'occupancy_status_id': (4, 2), 
                    'product_type_id': (2, 2), 
                    'property_type_id': (6, 2), 
                    'seller_id': (95, 3 + 10), 
                    'zip3_id': (1001, 3 + 20)
                },
                'emb_seq_dims': {
                    'yyyymm' : (219, 3 + 11),
                    'msa_id' : (407, 3 + 11),
                    'servicer_id' : (46, 3 + 3)
                },
                'lstm' : {
                    'lstm_size': (100 + 300),
                    'lstm_layers': (2 + 2),
                    'lstm_dropout': 0.23
                }
            },
            'pipe' : {
                'l1' : (50 + 50, 0.3),
                'l2' : (50 + 40, 0.4)
            }
        }
    }

    model = OTCS12Model(module_config)