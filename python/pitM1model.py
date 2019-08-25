
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

from fnm_macros_dset import FNMMacrosDataset, paddingCollator, load_data

class PitM12Model(nn.Module):
    r"""
     Fill up
    """
    def __init__(self, seq_n_features, lstm_size, lstm_layers, lstm_dropout, 
                    emb_acq_dims, emb_seq_dims):
        """
        Parameters:
            seq_n_features: number of features inside sequence. note that first
            19 (0-6 + 12) represent one hot encoding for DLQ
            lstm_size: the size of hidden unit in LSTM 
            linear_size: the number of of output units for the first linear
            embed_dims: the list of tuples, where the first element of tuple
                represents the number of categories and second element of the
                tuple if the embedding size
            embed_drp: drop out percentage after embedding layer
        """
        super(PitM12Model, self).__init__()
        self.emb_acq = nn.ModuleList([nn.Embedding(c, d) for c, d in emb_acq_dims])
        #Fix and add back
        #self.emb_seq = nn.ModuleList([nn.Embedding(c, d) for c, d in emb_seq_dims])

        # total dim from embeddings
        emb_acq_dim_sum = sum(x[1] for x in emb_acq_dims)
        # TODO: fix and add back
        #emb_seq_dim_sum = sum(x[1] for x in emb_seq_dims)
        emb_seq_dim_sum = 0
        #self.total_bn_dim = acq_bn_dim + seq_bn_dim
        #self.emb_bn = nn.BatchNorm1d(self.total_bn_dim)

        self.lstm_input_dim = seq_n_features + emb_acq_dim_sum + emb_seq_dim_sum
        self.lstm = nn.LSTM(
            input_size  = self.lstm_input_dim,
            hidden_size = lstm_size,
            num_layers  = lstm_layers,
            dropout     = lstm_dropout,
            batch_first  = True
        )

        lin1_size = 2 * lstm_size
        lin2_size = 2 * lin1_size
        lin3_size = lstm_size

        self.linear1 = nn.Linear(lstm_size, lin1_size)
        self.bn1 = nn.BatchNorm1d(lin1_size)
        self.dpout1 = nn.Dropout(0.3)

        self.linear2 = nn.Linear(lin1_size, lin2_size)
        self.bn2 = nn.BatchNorm1d(lin2_size)
        self.dpout2 = nn.Dropout(0.3)

        self.linear3 = nn.Linear(lin2_size, lin3_size)
        self.bn3 = nn.BatchNorm1d(lin3_size)
        self.dpout3 = nn.Dropout(0.3)        
        
        self.linear4 = nn.Linear(lin3_size, 19*12)
        #self.bn4 = nn.BatchNorm1d(19*12)
        #self.dpout1 = nn.Dropout(0.1)

        #self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, seq, seq_len, ymd, acq):
        # embed acq
        # acq = acq.long()
        ea = [embed(acq[:,i]) for i, embed in enumerate(self.emb_acq)]
        ea = torch.cat(ea,  1)
        #e = functional.relu(e)
        #e = self.embedding_bn(e)
        # reshape and replicate e to attach to each time index
        ea = ea.reshape(seq.shape[0], -1, ea.shape[1])\
            .expand(-1, seq.shape[1], -1)

        # embed ymd
        # TODO: fix indexes 200212 to 2 for examples then add back
        #ey = [embed(ymd[:,i]) for i, embed in enumerate(self.emb_seq)]
        #ey = torch.cat(ey,  1)

        # glue embeddings to each time index of the seq
        #s = torch.cat([seq, ea, ey], 2)
        s = torch.cat([seq, ea], 2)

        # process sequence
        self.lstm.flatten_parameters()
        #total_length = seq.size(1)
        packed_input = pack_padded_sequence(s, seq_len, 
                                            batch_first=True)
        _, (ht, _)  = self.lstm(packed_input)

        # ht is the final element of the sequence shape (hidden_size, 1)
        # we are predicting 19 DLQ x 12 months
        out = ht[-1, :, :]
        out = self.bn1(functional.relu(self.linear1(out)))
        out = self.dpout1(out)

        out = self.bn2(functional.relu(self.linear2(out)))
        out = self.dpout2(out)

        out = self.bn3(functional.relu(self.linear3(out)))
        #out = self.dpout2(out)

        out = self.linear4(out)

        out = out.reshape(-1, 19, 12) 
        #out = self.logsoftmax(out)
        return out

# def makeEmeddings(acq, embeding_dim=2):
#     # add 1 since NULLs are extracted as 0
#     acq_cat = np.max(acq[:, 1:], axis=0) + 1
#     emb = [(acq_cat[i], embeding_dim) for i in range(acq_cat.shape[0])]
#     return emb

class TrainingContext:
    def __init__(self, model_path, model, loss_function, optimizer, 
                trainDL, validDL, SAVE_EVERY_N_BATCHES,
                LOSS_EVERY_N_BATCHES):
        self.model_path = model_path
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.trainDL = trainDL
        self.validDL = validDL

        self.train_losses = []
        self.valid_losses = []

        self.SAVE_EVERY_N_BATCHES = SAVE_EVERY_N_BATCHES
        self.LOSS_EVERY_N_BATCHES = LOSS_EVERY_N_BATCHES
        
        self.device = torch.device("cpu")

    def useGPU(self, use=False):
        if use:
            self.device = torch.device(\
                "cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.loss_function = self.loss_function.to(self.device)
            if torch.cuda.device_count() > 1:
                print("Training on", torch.cuda.device_count(), "GPUs")
                self.model = nn.DataParallel(self.model)
        else:
            self.device = torch.device("cpu")

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
        if self.device.type=='cpu':
            module_dic = self.model.state_dict()
        else:
            module_dic = self.model.module.state_dict()
        torch.save({ \
            'epoch': epoch,
            'model_state_dict': module_dic, \
            'optimizer_state_dict': self.optimizer.state_dict(), \
            'train_losses' : self.train_losses, \
            'valid_losses' : self.valid_losses \
            }, self.model_path \
        )

    def trainStep(self, seq, seq_len, ymd, acq, target):
        self.model.zero_grad()

        seq = seq.to(self.device)
        ymd = ymd.to(self.device)
        acq = acq.to(self.device)
        seq_len = seq_len.to(self.device)
        target = target.to(self.device)

        target_hat = self.model(seq, seq_len, ymd, acq)
        loss = self.loss_function(target_hat, target)
        loss_item = loss.item()
        loss.backward()
        self.optimizer.step()

        return loss_item

    def trainLoop(self, epoch):
        self.model.train()
        tq = tqdm.tqdm(self.trainDL)
        losses = []
        for bidx, (seq, seq_len, ymd, acq, target) in enumerate(tq):
            tq.set_description('Train: %i' % bidx)
            
            loss_item = self.trainStep(seq, seq_len, ymd, acq, target)
            losses.append(loss_item)

            if bidx % self.LOSS_EVERY_N_BATCHES  == 0:
                mean_loss = np.mean(losses)/self.trainDL.batch_size
                tq.set_postfix(trainLoss = "{:.12f}".format(mean_loss))
                #writer.add_scalar('loss/training', mean_loss, epoch*bidx)

            if bidx % self.SAVE_EVERY_N_BATCHES == self.SAVE_EVERY_N_BATCHES - 1:
                self.saveModel(epoch)

        mean_loss = np.mean(losses)/self.trainDL.batch_size
        self.train_losses.append(mean_loss)
        return mean_loss

    def validStep(self, seq, seq_len, ymd, acq, target):
        seq = seq.to(self.device)
        ymd = ymd.to(self.device)
        acq = acq.to(self.device)
        seq_len = seq_len.to(self.device)
        target = target.to(self.device)

        target_hat = self.model(seq, seq_len, ymd, acq)
        loss = self.loss_function(target_hat, target)

        return loss.item()

    def validLoop(self, epoch):
        self.model.eval()
        losses = []
        with torch.no_grad():
            tq = tqdm.tqdm(self.validDL)
            for bidx, (seq, seq_len, ymd, acq, target) in enumerate(tq):
                tq.set_description('Valid: %i' % bidx)
                
                loss_item = self.validStep(seq, seq_len, ymd, acq, target)
                losses.append(loss_item)

                if bidx % self.LOSS_EVERY_N_BATCHES  == 0:
                    mean_loss = np.mean(losses)/self.validDL.batch_size
                    tq.set_postfix(validLoss = "{:.12f}".format(mean_loss))
                    #writer.add_scalar('loss/validation', mean_loss, epoch*bidx)

        mean_loss = np.mean(losses)/self.validDL.batch_size
        self.valid_losses.append(mean_loss)
        return mean_loss
        


def makeModel(params):
    model = PitM12Model(
        seq_n_features = model_params['seq_n_features'], 
        lstm_size = model_params['lstm_size'], 
        lstm_layers = model_params['lstm_layers'], 
        lstm_dropout = model_params['lstm_dropout'], 
        emb_acq_dims = model_params['emb_acq_dims'], 
        emb_seq_dims = model_params['emb_seq_dims']
    )
    return model


if __name__ == "__main__":
    TRAIN_PATH = '/home/user/notebooks/data/train'
    VALID_PATH = '/home/user/notebooks/data/valid'
    MODEL_PATH = '/home/user/notebooks/data/model/pitM1'
    SUMMARY_PATH = '/home/user/notebooks/runs/pitM1'
    MODEL_SAVED = MODEL_PATH + '/pitM1.pth'

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    DEBUG = False

    if DEBUG:
        BATCH_SIZE = 40
        NUM_WORKERS = 0
        LOSS_EVERY_N_BATCHES=10
        SAVE_EVERY_N_BATCHES=1000
    else:
        BATCH_SIZE = 512 #12*2**10
        NUM_WORKERS = 8
        LOSS_EVERY_N_BATCHES=100
        SAVE_EVERY_N_BATCHES=1000

    NUM_EPOCHS = 100
    
    t_acq, t_idx_to_seq, t_seq, t_macros, t_ym2idx = load_data(TRAIN_PATH, True, False)
    v_acq, v_idx_to_seq, v_seq, v_macros, v_ym2idx = load_data(VALID_PATH, True, False)

    train_ds = FNMMacrosDataset(t_acq, t_idx_to_seq, t_seq, t_macros, t_ym2idx, 12, 10)
    valid_ds = FNMMacrosDataset(v_acq, v_idx_to_seq, v_seq, v_macros, v_ym2idx, 12, 10)

    print("Number of train acq: {:,}".format(len(train_ds)))
    print("Number of valid acq: {:,}".format(len(valid_ds)))
    
    trainDL = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle = True, \
        collate_fn=paddingCollator, num_workers=NUM_WORKERS)
    validDL = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle = True, \
        collate_fn=paddingCollator, num_workers=NUM_WORKERS)

    model_params = {
        'seq_n_features': 39, # seq + macros
        'lstm_size': 250,
        'lstm_layers': 4,
        'lstm_dropout': 0.2,
        'emb_acq_dims' : [(55, 2), # state id
                          (5, 2), # purpose_id
                          (4, 2), # mi_type_id
                          (4, 2), # occupancy_status_id
                          (2, 2), # product_type_id
                          (6, 2), # property_type_id
                          (95, 2), # seller_id
                          (1001, 3)], # zip3
        'emb_seq_dims' : [(219, 2), # yyyymm
                          (46, 2), # servicer
                          (49741, 2)] #msa
    }

    model = makeModel(model_params)
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(params=model.parameters())

    fitCtx = TrainingContext(
        MODEL_SAVED, 
        model, 
        loss_function, 
        optimizer, 
        trainDL, 
        validDL,
        SAVE_EVERY_N_BATCHES,
        LOSS_EVERY_N_BATCHES
    )

    checkpoint_epoch = fitCtx.loadModel()

    fitCtx.useGPU(not DEBUG)

    #writer = SummaryWriter(SUMMARY_PATH, comment='fixed_seq_ind_2')

    try:
        with tqdm.trange(checkpoint_epoch, checkpoint_epoch + NUM_EPOCHS) as t:
            for epoch in t:
                t.set_description('Epoch: %i' % epoch)
                train_loss = fitCtx.trainLoop(epoch)
                #train_loss = 0
                valid_loss = fitCtx.validLoop(epoch)

                t.set_postfix(
                    trainLoss = "{:.12f}".format(train_loss),
                    validLoss = "{:.12f}".format(valid_loss)
                )

                fitCtx.saveModel(epoch)

    except KeyboardInterrupt:
        print ('Saving the model state before exiting')
        fitCtx.saveModel(epoch)
    except RuntimeError as e:
        print(e)
    #writer.close()
