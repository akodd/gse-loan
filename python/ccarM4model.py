
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

from fnm_ccar_dset import FNMCCARDataset, paddingCollator, load_data

class LinearBlock(nn.Module):
    def __init__(self, input_size, linear_conf):
        super(LinearBlock, self).__init__()
        dc = collections.OrderedDict()
        prev_s = input_size
        for i, (s, r) in enumerate(linear_conf):
            dc['linear'+str(i+1)]=nn.Linear(prev_s, s)
            dc['relu'+str(i+1)] = nn.ReLU(inplace=True)
            dc['batchn'+str(i+1)] = nn.BatchNorm1d(s)
            dc['drpout'+str(i+1)] = nn.Dropout(r)
            prev_s = s
        self.linblock = nn.Sequential(dc)

    def forward(self, x):
        return self.linblock(x)

class MacroMortEcoder(nn.Module):
    def __init__(self, seq_n_features, lstm_conf, emb_acq_dims, emb_seq_dims, linear_conf):
        super(MacroMortEcoder, self).__init__()
        emb_acq_dict = collections.OrderedDict()
        emb_acq_dim_sum = 0
        for name, c, d in emb_acq_dims:
            emb_acq_dict[name] = nn.Embedding(c, d)
            emb_acq_dim_sum += d
        self.emb_acq = nn.ModuleDict(emb_acq_dict)

        emb_seq_dict = collections.OrderedDict()
        emb_seq_dim_sum = 0
        for name, c, d in emb_seq_dims:
            emb_seq_dict[name] = nn.Embedding(c, d)
            emb_seq_dim_sum += d
        self.emb_seq = nn.ModuleDict(emb_seq_dict)
       
        emb_dim_total = emb_acq_dim_sum + emb_seq_dim_sum

        lstm_size = lstm_conf['lstm_size']
        lstm_layers = lstm_conf['lstm_layers']
        lstm_dropout = lstm_conf['lstm_dropout']

        self.lstm = nn.LSTM(    
            input_size  = seq_n_features + emb_dim_total,
            hidden_size = lstm_size,
            num_layers  = lstm_layers,
            dropout     = lstm_dropout,
            batch_first  = True,
            bidirectional = True
        )

        #lin_block_input = (1 + self.lstm.bidirectional) * \
        #    self.lstm.num_layers * self.lstm.hidden_size

        self.linblock = LinearBlock(self.lstm.hidden_size, linear_conf)

    def forward(self, seq, seq_len, ymd, acq):
        ea = [x(acq[:,i]) for i, (a, x) in enumerate(self.emb_acq.items())]
        ea = torch.cat(ea,  1)
        ea = ea.reshape(seq.shape[0], -1, ea.shape[1])\
            .expand(-1, seq.shape[1], -1)

        ey = [x(ymd[:,:,i]) for i, (a, x) in enumerate(self.emb_seq.items())]
        ey = torch.cat(ey,  2)

        s = torch.cat([seq, ea, ey], 2)

        self.lstm.flatten_parameters()
        packed_input = pack_padded_sequence(s, seq_len, batch_first=True)
        _, (ht, _)  = self.lstm(packed_input)

        # move batch from dim 1 to 0
        #out = ht.permute(1, 0, 2).view(-1, 
        #    2 * self.lstm.num_layers * self.lstm.hidden_size)

        out = ht.view(self.lstm.num_layers, 2, -1, self.lstm.hidden_size)
        out = out[-1, 0, :, :] + out[-1, 1, :, :]

        out = self.linblock(out)
        # middle = int(out.shape[1]/2)
        # m = out[:,       :middle]
        # s = out[:, middle:]
        # return m,s
        batch_idx = torch.arange(seq.shape[0], device=seq.device)
        dlq = seq[batch_idx, seq_len.long()-1, -9:]
        return out, dlq

class MacroMortDecoder(nn.Module):
    def __init__(self, input_size, lstm_conf, pre_lin_conf, post_lin_conf):
        super(MacroMortDecoder, self).__init__()

        self.pre_linblock = LinearBlock(input_size, pre_lin_conf)
        self.lstm = nn.LSTM(    
            input_size  = lstm_conf['input_size'],
            hidden_size = lstm_conf['lstm_size'],
            num_layers  = lstm_conf['lstm_layers'],
            dropout     = lstm_conf['lstm_dropout'],
            batch_first  = False,
            bidirectional = False
        )
        
        #post_input_size = (1 + self.lstm.bidirectional) * \
        #    self.lstm.num_layers * self.lstm.hidden_size
        self.post_linblock = LinearBlock(self.lstm.hidden_size+9, post_lin_conf)

    def forward(self, zim, dlq, macro):
        self.lstm.flatten_parameters()
        zim = self.pre_linblock(zim)

        out = []
        for t in range(12):
            decinp = torch.cat([zim, dlq, macro[:, t, :]], 1)
            decinp = decinp.unsqueeze(0) # sequence length is 1
            _, (hx, _) = self.lstm(decinp)

            lstm_out = hx[-1, :, :]
            lstm_out = torch.cat([dlq, lstm_out], 1)
            dlq_dist = self.post_linblock(lstm_out)
            out.append(dlq_dist)
            # batch, 9
            _, idx = F.softmax(dlq_dist, 1).max(1)
            dlq = torch.zeros_like(dlq)
            dlq[torch.arange(dlq.shape[0]), idx]=1

        dlq_seq = torch.stack(out, 2) # (batch, 9[0,...,7,EOS], 12)

        return dlq_seq

class CCARM4Model(nn.Module):
    r"""
     Fill up
    """
    def __init__(self, macroMortEcoder, macroMortDecoder):
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
        super(CCARM4Model, self).__init__()
        self.encoder = macroMortEcoder
        self.decoder = macroMortDecoder    


    def forward(self, seq, seq_len, ymd, acq, macro_pred):
        out, dlq = self.encoder(seq, seq_len, ymd, acq)
        dlq_seq = self.decoder(out, dlq, macro_pred)
        return dlq_seq


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class Losses:
    def __init__(self, patience=10):
        self.train_losses = []
        self.valid_losses = []
        self.test_losses  = []

    def addValidLoss(self, item):
        self.valid_losses.append(item)
        


class TrainingContext:
    def __init__(self, model_path, model, loss_function, optimizer, 
                trainDL, validDL, SAVE_EVERY,
                PRINT_EVERY):
        self.model_path = model_path
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.trainDL = trainDL
        self.validDL = validDL

        self.train_losses = []
        self.valid_losses = []

        self.SAVE_EVERY = SAVE_EVERY
        self.PRINT_EVERY = PRINT_EVERY
        
        self.device = torch.device("cpu")

    def trainStep(self, seq, seq_len, ymd, acq, macro_pred, target):
        self.model.zero_grad()

        target_hat = self.model(seq, seq_len, ymd, acq, macro_pred)
        target = target.to(target_hat.device)
        loss = self.loss_function(target_hat, target)
        loss_item = loss.item()
        loss.backward()
        self.optimizer.step()

        return loss_item

    def trainLoop(self, epoch):
        self.model.train()
        tq = tqdm.tqdm(self.trainDL)
        losses = []
        for bidx, (seq, seq_len, ymd, acq, macro_pred, target) in enumerate(tq):
            tq.set_description('Train: %i' % bidx)
            
            loss_item = self.trainStep(seq, seq_len, ymd, acq, macro_pred, target)
            losses.append(loss_item)

            if bidx % self.PRINT_EVERY  == 0:
                mean_loss = np.mean(losses)/self.trainDL.batch_size
                tq.set_postfix(trainLoss = "{:.4f}".format(mean_loss))
                #writer.add_scalar('loss/training', mean_loss, epoch*bidx)

        mean_loss = np.mean(losses)/self.trainDL.batch_size
        self.train_losses.append(mean_loss)
        return mean_loss

    def validStep(self, seq, seq_len, ymd, acq, macro_pred, target):
        target_hat = self.model(seq, seq_len, ymd, acq, macro_pred)
        target = target.to(target_hat.device)
        loss = self.loss_function(target_hat, target)
        return loss.item()

    def validLoop(self, epoch):
        self.model.eval()
        losses = []
        with torch.no_grad():
            tq = tqdm.tqdm(self.validDL)
            for bidx, (seq, seq_len, ymd, acq, macro_pred, target) in enumerate(tq):
                tq.set_description('Valid: %i' % bidx)
                
                loss_item = self.validStep(seq, seq_len, ymd, acq, macro_pred, target)
                losses.append(loss_item)

                if bidx % self.PRINT_EVERY  == 0:
                    mean_loss = np.mean(losses)/self.validDL.batch_size
                    tq.set_postfix(validLoss = "{:.4f}".format(mean_loss))
                    #writer.add_scalar('loss/validation', mean_loss, epoch*bidx)

        mean_loss = np.mean(losses)/self.validDL.batch_size
        self.valid_losses.append(mean_loss)
        return mean_loss
        

    def useGPU(self, use=False):
        if use:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        print('Using device: {}'.format(self.device.type))
        self.model.to(self.device)
        #self.loss_function = self.loss_function.to(self.device)


    def loadModel(self):
        if os.path.exists(self.model_path):
            print('Loading model checkpoint: {}'.format(self.model_path))
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint['train_losses']
            self.valid_losses = checkpoint['valid_losses']
            checkpoint_epoch = checkpoint['epoch']
        else:
            checkpoint_epoch = 0
        return checkpoint_epoch

    def saveModel(self, epoch):
        torch.save({ \
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(), \
            'optimizer_state_dict': self.optimizer.state_dict(), \
            'train_losses' : self.train_losses, \
            'valid_losses' : self.valid_losses \
            }, self.model_path \
        ) 

    def makeParallel(self, use=False):
        if use and torch.cuda.device_count() > 1:
            print("Training on", torch.cuda.device_count(), "GPUs")
            self.model = nn.DataParallel(self.model)

def makeModel(model_params):
    encoder = MacroMortEcoder(
        seq_n_features = model_params['seq_n_features'],
        lstm_conf = model_params['encoder']['lstm'],
        emb_acq_dims = model_params['encoder']['emb_acq_dims'],
        emb_seq_dims = model_params['encoder']['emb_seq_dims'],
        linear_conf = model_params['encoder']['lin_block'],
    )

    decoder = MacroMortDecoder(
        input_size = model_params['decoder']['input_size'],
        lstm_conf = model_params['decoder']['lstm'],
        pre_lin_conf = model_params['decoder']['pre_lin_conf'],
        post_lin_conf = model_params['decoder']['post_lin_conf'],
    )

    model = CCARM4Model(encoder, decoder)
    return model

def adjustModelPath(model_path, restart=True):
    if not restart:
        est = timezone('EST')
        model_path = model_path + '/' + datetime.datetime.now(est)\
            .strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path


if __name__ == "__main__":
    TRAIN_PATH = '/home/user/notebooks/data/train'
    VALID_PATH = '/home/user/notebooks/data/valid'
    SUMMARY_PATH = '/home/user/notebooks/runs/ccarM3'
    
    # run specific model path
    MODEL_PATH = adjustModelPath(
        '/home/user/notebooks/data/model/ccarM3/', 
        restart=False
    )
    print('Model will be saved in: '+MODEL_PATH)
    MODEL_SAVED = MODEL_PATH + '/ccarM3.pth'

    DEBUG = False

    if DEBUG:
        BATCH_SIZE = 3
        NUM_WORKERS = 0
        PRINT_EVERY=10
        SAVE_EVERY=1000
    else:
        BATCH_SIZE = 512 
        NUM_WORKERS = 6
        PRINT_EVERY=100
        SAVE_EVERY=1000

    NUM_EPOCHS = 250
    
    t_acq, t_idx_to_seq, t_seq, t_macros, t_ym2idx = load_data(TRAIN_PATH, 
        verbose=True, oneChunkOnly=True)
    v_acq, v_idx_to_seq, v_seq, v_macros, v_ym2idx = load_data(VALID_PATH, 
        verbose=True, oneChunkOnly=True)

    train_ds = FNMCCARDataset(t_acq, t_idx_to_seq, t_seq, t_macros, t_ym2idx, 12, 1)
    valid_ds = FNMCCARDataset(v_acq, v_idx_to_seq, v_seq, v_macros, v_ym2idx, 12, 1)

    print("Number of train acq: {:,}".format(len(train_ds)))
    print("Number of valid acq: {:,}".format(len(valid_ds)))
    
    trainDL = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle = True, \
        collate_fn=paddingCollator, num_workers=NUM_WORKERS, pin_memory=True)
    validDL = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle = True, \
        collate_fn=paddingCollator, num_workers=NUM_WORKERS, pin_memory=True)

    model_params = {
        'seq_n_features': FNMCCARDataset.seq_n_features,
        'encoder' : {
            'emb_acq_dims' : [('state_id',   55, 25), 
                            ('purpose_id', 5, 2),
                            ('mi_type_id', 4, 2),
                            ('occupancy_status_id', 4, 2), 
                            ('product_type_id', 2, 2), 
                            ('property_type_id', 6, 2), 
                            ('seller_id', 95, 40), 
                            ('zip3_id', 1001, 50)],
            'emb_seq_dims' : [('yyyymm', 219, 50), 
                            ('msa_id', 407, 50), 
                            ('servicer_id', 46, 23)],
            'lstm' : {
                'lstm_size': 400,
                'lstm_layers': 3,
                'lstm_dropout': 0.2
            },
            'lin_block' : [
                (200, 0.2),
                (100, 0.2),
                (50,  0.2)
            ]
        },
        'decoder' : {
            'input_size': 50,
            'pre_lin_conf': [
                (50, 0.2)
            ],
            'lstm' : {
                'input_size' : 73, # 50 + 9 + macros
                'lstm_size': 100,
                'lstm_layers': 3,
                'lstm_dropout': 0.2
            },
            'post_lin_conf' : [
                (75, 0.1),
                (50, 0.1),
                (9, 0.1) # DLQ is from 0 to 8
            ]
        }
    }

    model = makeModel(model_params)
    loss_function = nn.CrossEntropyLoss(reduction='sum',  ignore_index=8) # 8 is EOS
    optimizer = optim.Adam(params=model.parameters())

    fitCtx = TrainingContext(
        MODEL_SAVED, 
        model, 
        loss_function, 
        optimizer, 
        trainDL, 
        validDL,
        SAVE_EVERY,
        PRINT_EVERY
    )

    fitCtx.useGPU(not DEBUG)
    checkpoint_epoch = fitCtx.loadModel()
    fitCtx.makeParallel(not DEBUG)

    print(fitCtx.model)
    print(fitCtx.loss_function)

    #writer = SummaryWriter(SUMMARY_PATH, comment='fixed_seq_ind_2')

    def lprint(x):
        return "|".join(map(lambda x: "{:.2f}".format(x), x))

    try:
        with tqdm.trange(checkpoint_epoch, checkpoint_epoch + NUM_EPOCHS) as t:
            for epoch in t:
                t.set_description('Epoch: %i' % epoch)
                train_loss = fitCtx.trainLoop(epoch)

                t.set_postfix(
                    TL = "{:.2f}".format(train_loss), 
                    VL = lprint(fitCtx.valid_losses[-10:])
                )

                valid_loss = fitCtx.validLoop(epoch)

                t.set_postfix(
                    TL = "{:.2f}".format(train_loss), 
                    VL = lprint(fitCtx.valid_losses[-10:])
                )

                fitCtx.saveModel(epoch)

    except KeyboardInterrupt:
        print ('Saving the model state before exiting')
        fitCtx.saveModel(epoch)
    #except RuntimeError as e:
    #    print(e)
    #writer.close()
