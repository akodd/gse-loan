
import torch as torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import glob
import numpy as np
import tqdm
import os
import time
import datetime
from pytz import timezone

from fnm_ccar_dset import FNMCCARDataset, paddingCollator, load_data

class CCARM1Model(nn.Module):
    r"""
     Fill up
    """
    def __init__(self, seq_n_features, lstm_size, lstm_layers, lstm_dropout, 
                    emb_acq_dims, emb_seq_dims, latten_size=None):
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
        super(CCARM1Model, self).__init__()
        self.emb_acq = nn.ModuleList([nn.Embedding(c, d) for c, d in emb_acq_dims])
        #Fix and add back
        self.emb_seq = nn.ModuleList([nn.Embedding(c, d) for c, d in emb_seq_dims])

        # total dim from embeddings
        emb_acq_dim_sum = sum(x[1] for x in emb_acq_dims)
        emb_seq_dim_sum = sum(x[1] for x in emb_seq_dims)
        #emb_seq_dim_sum = 0
        #self.total_bn_dim = acq_bn_dim + seq_bn_dim
        #self.emb_bn = nn.BatchNorm1d(self.total_bn_dim)

        self.lstm_input_dim = seq_n_features + emb_acq_dim_sum + emb_seq_dim_sum
        self.lstm = nn.LSTM(    
            input_size  = self.lstm_input_dim,
            hidden_size = lstm_size,
            num_layers  = lstm_layers,
            dropout     = lstm_dropout,
            batch_first  = True,
            bidirectional = True
        )

        if latten_size is None:
            latten_size = lstm_size
        self.latent_lin = nn.Linear(lstm_size, 2*latten_size) # m and s

        self.lstm_dec_cell1 = nn.LSTMCell(
            input_size=latten_size + 9 + 14, # 9 for DLQ + 14 macro vars
            hidden_size=100
        )

        self.lstm_dec_dpout = nn.Dropout(0.1)

        self.lstm_dec_cell2 = nn.LSTMCell(
            input_size=100, # 9 for DLQ + 14 macro vars
            hidden_size=100
        )

        # output is 9 for 0,...,7,8 DLQ
        self.gen_dlq_dist1 = nn.Linear(self.lstm_dec_cell2.hidden_size, 25)
        self.bn1 = nn.BatchNorm1d(25)
        self.dpout1 = nn.Dropout(0.1)

        self.gen_dlq_dist2 = nn.Linear(25, 9)

    def forward(self, seq, seq_len, ymd, acq, macro_pred):
        # embed acq
        # acq = acq.long()
        ea = [embed(acq[:,i]) for i, embed in enumerate(self.emb_acq)]
        ea = torch.cat(ea,  1)
        #ea = functional.relu(ea)
        #ea = self.embedding_bn(ea)
        # reshape and replicate e to attach to each time index
        ea = ea.reshape(seq.shape[0], -1, ea.shape[1])\
            .expand(-1, seq.shape[1], -1)

        # embed ymd
        ey = [embed(ymd[:,:,i]) for i, embed in enumerate(self.emb_seq)]
        ey = torch.cat(ey,  2)

        # glue embeddings to each time index of the seq
        s = torch.cat([seq, ea, ey], 2)

        # process sequence
        self.lstm.flatten_parameters()
        #total_length = seq.size(1)
        packed_input = pack_padded_sequence(s, seq_len, batch_first=True)
        _, (ht, _)  = self.lstm(packed_input)

        # ht is the final element of the sequence shape (hidden_size, 1)
        #output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        lat = ht.view(self.lstm.num_layers, 2, -1, self.lstm.hidden_size)
        lat = lat[-1, 0, :, :] + lat[-1, 1, :, :] #ht[-1, :, :]
        lat = self.latent_lin(lat)

        # split ht into mean and sd parts
        m = lat[:,                     :int(lat.shape[1]/2)]
        s = lat[:, int(lat.shape[1]/2):]
        # sample
        z = torch.randn(m.shape, device=m.device)
        z = m + torch.sqrt(torch.exp(s))*z

        # we are predicting sequences for 12 months into the future

        h1x = torch.zeros([z.shape[0], self.lstm_dec_cell1.hidden_size], device=z.device)
        c1x = torch.zeros_like(h1x)
        h2x = torch.zeros([z.shape[0], self.lstm_dec_cell2.hidden_size], device=z.device)
        c2x = torch.zeros_like(h2x)
        #dec_input = z # initial hidden state from sample
        # last DLQ is realized so one-hot is correct
        dlq_dist = seq[torch.arange(seq.shape[0], device=z.device), seq_len.long()-1, -9:]

        out = []
        for t in range(12):
            decinp = torch.cat([z, dlq_dist, macro_pred[:, t, :]], 1)
            h1x, c1x = self.lstm_dec_cell1(decinp, (h1x, c1x))
            h2_in = self.lstm_dec_dpout(h1x)
            h2x, c2x = self.lstm_dec_cell2(h2_in, (h2x, c2x))
            dlq_dist = self.bn1(functional.relu(self.gen_dlq_dist1(h2x)))
            dlq_dist = self.dpout1(dlq_dist)
            dlq_dist = self.gen_dlq_dist2(dlq_dist)
            out.append(dlq_dist)

        dlq_seq = torch.stack(out, 2) # (batch, 9[0,...,7,EOS], 12)

        return dlq_seq



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
    model = CCARM1Model(
        seq_n_features = model_params['seq_n_features'], 
        lstm_size = model_params['lstm_size'], 
        lstm_layers = model_params['lstm_layers'], 
        lstm_dropout = model_params['lstm_dropout'], 
        emb_acq_dims = model_params['emb_acq_dims'], 
        emb_seq_dims = model_params['emb_seq_dims']
    )
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
    SUMMARY_PATH = '/home/user/notebooks/runs/ccarM1'
    
    # run specific model path
    MODEL_PATH = adjustModelPath('/home/user/notebooks/data/model/ccarM1', False);
    print('Model will be saved in: '+MODEL_PATH)
    MODEL_SAVED = MODEL_PATH + '/ccarM1.pth'

    DEBUG = True

    if DEBUG:
        BATCH_SIZE = 3
        NUM_WORKERS = 0
        PRINT_EVERY=10
        SAVE_EVERY=1000
    else:
        BATCH_SIZE = 256 
        NUM_WORKERS = 6
        PRINT_EVERY=100
        SAVE_EVERY=1000

    NUM_EPOCHS = 5
    
    t_acq, t_idx_to_seq, t_seq, t_macros, t_ym2idx = load_data(TRAIN_PATH, True, True)
    v_acq, v_idx_to_seq, v_seq, v_macros, v_ym2idx = load_data(VALID_PATH, True, True)

    train_ds = FNMCCARDataset(t_acq, t_idx_to_seq, t_seq, t_macros, t_ym2idx, 12, 0)
    valid_ds = FNMCCARDataset(v_acq, v_idx_to_seq, v_seq, v_macros, v_ym2idx, 12, 0)

    print("Number of train acq: {:,}".format(len(train_ds)))
    print("Number of valid acq: {:,}".format(len(valid_ds)))
    
    trainDL = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle = True, \
        collate_fn=paddingCollator, num_workers=NUM_WORKERS, pin_memory=True)
    validDL = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle = True, \
        collate_fn=paddingCollator, num_workers=NUM_WORKERS, pin_memory=True)

    model_params = {
        'seq_n_features': 29, # seq + macros
        'lstm_size': 300,
        'lstm_layers': 5,
        'lstm_dropout': 0.2,
        'emb_acq_dims' : [(55, 25), # state id
                          (5, 2), # purpose_id
                          (4, 2), # mi_type_id
                          (4, 2), # occupancy_status_id
                          (2, 2), # product_type_id
                          (6, 2), # property_type_id
                          (95, 40), # seller_id
                          (1001, 50)], # zip3
        'emb_seq_dims' : [(219, 50), # yyyymm
                          (407, 50), #msa
                          (46, 23)] # servicer
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

    try:
        with tqdm.trange(checkpoint_epoch, checkpoint_epoch + NUM_EPOCHS) as t:
            for epoch in t:
                t.set_description('Epoch: %i' % epoch)
                train_loss = fitCtx.trainLoop(epoch)
                valid_loss = fitCtx.validLoop(epoch)

                t.set_postfix(
                    trainLoss = "{:.4f}".format(train_loss),
                    validLoss = "{:.4f}".format(valid_loss)
                )

                fitCtx.saveModel(epoch)

    except KeyboardInterrupt:
        print ('Saving the model state before exiting')
        fitCtx.saveModel(epoch)
    #except RuntimeError as e:
    #    print(e)
    #writer.close()
