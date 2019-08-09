import torch as torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import *
import numpy as np
from os import path
import glob
from torch.nn.utils.rnn import *
from baselM1model import *
import tqdm

class FNMDataset(Dataset):

    def __init__(self, acq, idx_to_seq, seq, ratio = 0):
        self.acq = acq
        self.seq = seq
        self.idx_to_seq = idx_to_seq
        if ratio > 0:
            defaulted = acq[:,0]==1
            n_defaulted = np.sum(defaulted)
            def_idx = np.random.choice(np.where(defaulted)[0], n_defaulted, False)
            nondef_idx = np.random.choice(np.where(~defaulted)[0], int(n_defaulted*ratio) , False)
            sub_idx = np.concatenate([def_idx, nondef_idx])
            self.length = sub_idx.shape[0]
            self.idx_reduction = sub_idx
        else:
            self.length = acq.shape[0]
            self.idx_reduction = np.arange(self.length)

    def __len__(self):
        return self.length
        #return 100
    
    def __getitem__(self, itemID):
        idx = self.idx_reduction[itemID]
        account = self.acq[idx, :]
        seq_info = self.idx_to_seq[idx, :]
        chunk_num = seq_info[0]
        seq_idx_begin = seq_info[1]
        seq_idx_end = seq_info[2]
        # sequence columns are:
        # 'default_1y', 'yyyymm', 'dlq', 'age', 'int_rate', 'current_upb'
        #sequence = self.seq[0][seq_idx_begin:seq_idx_end, :] #TODO change back
        sequence = self.seq[chunk_num][seq_idx_begin:seq_idx_end, :]
        sequence = sequence[sequence[:,2] <= 7, :] # filter DLQ above 7
        default_1y = sequence[:, 0]
        dlq = sequence[:, 2].astype(int)
        dlq = np.eye(8, dtype=np.float32)[dlq] # dlq is between 0 and 7
        sequence = np.concatenate([dlq, sequence[:, 3:]], axis=1)

        return sequence, account, default_1y

def paddingCollator(batch):
    seq_batch = [torch.from_numpy(batch[i][0]) for i in range(len(batch))]

    seq_lengths = [seq_batch[b].shape[0] for b in range(len(seq_batch))]
    seq_lengths, seq_perm_idx = torch.Tensor(seq_lengths).int().sort(0, descending=True)

    seq_batch = pad_sequence(seq_batch, batch_first=True)
    seq_batch = seq_batch[seq_perm_idx, :, :]

    acq_batch = [torch.from_numpy(batch[i][1]) for i in range(len(batch))]
    acq_batch = torch.stack(acq_batch)
    acq_batch = acq_batch[seq_perm_idx, :]

    default_1y_batch = [torch.from_numpy(batch[i][2]) for i in range(len(batch))]
    default_1y_batch = pad_sequence(default_1y_batch, batch_first=True)
    default_1y_batch = default_1y_batch[seq_perm_idx, :]

    return seq_batch, seq_lengths, acq_batch, default_1y_batch

def load_data(data_path):
    acquisition_nname = data_path + '/fnm_input_acq.npy'
    sequence_nname = data_path + '/fnm_input_seq_*.npy' #TODO switch back to *
    idx_to_seq_nname = data_path + '/fnm_input_idx_to_seq.npy'
    
    print('Acquisition numpy: {}'.format(acquisition_nname))
    print('Sequence numpy: {}'.format(sequence_nname))
    print('Index to Sequence Index numpy: {}'.format(idx_to_seq_nname))
    
    seq_files = sorted([f for f in glob.glob(sequence_nname, recursive=False)])
    seq_numpy = [None] * len(seq_files)
    
    acq_numpy = np.load(acquisition_nname)
    idx_to_seq = np.load(idx_to_seq_nname)
    
    for chunk_idx, seq_numpy_chunk in enumerate(seq_files):
        print("loading file: {}".format(seq_numpy_chunk))
        seq_numpy[chunk_idx] = np.load(seq_numpy_chunk)
        
    return acq_numpy, idx_to_seq, seq_numpy

def weightedLoss(default_1y, default_1y_hat, seq_lengths):
    weights = torch.zeros_like(default_1y_hat)
    for b in range(default_1y.size(0)):
        weights[b, :seq_lengths[b]] = 1.0

    loss = nn.functional.binary_cross_entropy(\
        default_1y_hat, \
        default_1y,  weight=weights, \
        reduction = 'mean' \
    )
    return loss

if __name__ == "__main__":
    TRAIN_PATH = '/home/user/notebooks/data/train'
    VALID_PATH = '/home/user/notebooks/data/valid'
    MODEL_PATH = '/home/user/notebooks/data/model/baselm1'
    MODEL_SAVED = MODEL_PATH + '/baselm1.pth'

    train_acq_numpy, train_idx_to_seq, train_seq_numpy = load_data(TRAIN_PATH)
    valid_acq_numpy, valid_idx_to_seq, valid_seq_numpy = load_data(VALID_PATH)
    
    train_ds = FNMDataset(train_acq_numpy, train_idx_to_seq, train_seq_numpy, 1.5)
    valid_ds = FNMDataset(valid_acq_numpy, valid_idx_to_seq, valid_seq_numpy)

    print("Number of train acq: {:,}".format(len(train_ds)))
    print("Number of valid acq: {:,}".format(len(valid_ds)))
    

    BATCH_SIZE = 8000 # TODO: change back to 10000
    NUM_WORKERS = 5 # change back to 10 TODO:
    BATCH_STOP = 100
    NUM_EPOCHS = 100
    LOSS_EVERY_N_BATCHES=10
    SAVE_EVERY_N_BATCHES=100

    trainDL = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle = True, \
        collate_fn=paddingCollator, num_workers=NUM_WORKERS)
    validDL = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle = True, \
        collate_fn=paddingCollator, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu') #TODO: change back to cuda

    model = BaselM1Model(11, 50, 5).to(device)
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.RMSprop(params=model.parameters())
    checkpoint_epoch = 0
    train_losses = []
    valid_losses = []
    train_preds = 0.0
    valid_preds = 0.0

    if path.exists(MODEL_SAVED):
        print('Loading model checkpoint: {}'.format(MODEL_SAVED))
        checkpoint = torch.load(MODEL_SAVED)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        train_preds = checkpoint['train_preds']
        valid_losses = checkpoint['valid_losses']
        valid_preds = checkpoint['valid_preds']

    if torch.cuda.device_count() > 1:
        print("Training on", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    try:
        with tqdm.trange(checkpoint_epoch, checkpoint_epoch + NUM_EPOCHS) as t:
            for epoch in t:
                train_losses_epoch = []
                t.set_description('Epoch: %i' % epoch)
                model.train()
                tqdm_inner = tqdm.tqdm(trainDL)
                for bidx, (seq, seq_len, acq, def1y) in enumerate(tqdm_inner):
                    tqdm_inner.set_description('Train: %i' % bidx)
                    model.zero_grad()
                    seq = seq.to(device)
                    acq = acq.to(device)
                    seq_len = seq_len.to(device)
                    def1y = def1y.to(device)

                    def1y_hat = model(seq, acq, seq_len, def1y)
                    loss = weightedLoss(def1y, def1y_hat, seq_len)
                    train_losses_epoch.append(loss.item())
                    train_preds +=torch.sum(seq_len).item()
                    loss.backward()
                    optimizer.step()

                    if bidx % LOSS_EVERY_N_BATCHES  == 0:
                        tqdm_inner.set_postfix( \
                            trainLoss = "{:.12f}".format(np.mean(train_losses_epoch)), \
                            train_preds=train_preds)

                    if bidx % SAVE_EVERY_N_BATCHES == 0 and bidx > 0:
                        torch.save({ \
                            'epoch': epoch,
                            'model_state_dict': model.module.state_dict(), \
                            'optimizer_state_dict': optimizer.state_dict(), \
                            'train_losses' : train_losses, \
                            'valid_losses' : valid_losses, \
                            'train_preds' : train_preds, \
                            'valid_preds' : valid_preds \
                            }, MODEL_SAVED \
                        )

                model.eval()
                with torch.no_grad():
                    valid_losses_epoch = []
                    tqdm_inner = tqdm.tqdm(validDL)
                    for bidx, (seq, seq_len, acq, def1y) in enumerate(tqdm_inner):
                        tqdm_inner.set_description('Valid: %i' % bidx)
                        seq = seq.to(device)
                        acq = acq.to(device)
                        seq_len = seq_len.to(device)
                        def1y = def1y.to(device)
                        def1y_hat = model(seq, acq, seq_len, def1y)
                        loss = weightedLoss(def1y, def1y_hat, seq_len)
                        valid_losses_epoch.append(loss.item())
                        valid_preds +=torch.sum(seq_len).item()

                        if bidx % LOSS_EVERY_N_BATCHES  == 0:
                            tqdm_inner.set_postfix( \
                                validLoss = "{:.12f}".format(np.mean(valid_losses_epoch)), \
                                valid_preds = valid_preds)

                t.set_postfix(\
                    trainLoss = "{:.12f}".format(np.mean(train_losses_epoch)), \
                    validLoss = "{:.12f}".format(np.mean(valid_losses_epoch)))

                # save losses and model
                torch.save({ \
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(), \
                    'optimizer_state_dict': optimizer.state_dict(), \
                    'train_losses' : train_losses, \
                    'valid_losses' : valid_losses, \
                    'train_preds' : train_preds, \
                    'valid_preds' : valid_preds \
                    }, MODEL_SAVED \
                )

                train_losses.append(np.mean(train_losses_epoch))
                valid_losses.append(np.mean(valid_losses_epoch))

    except KeyboardInterrupt:
        print ('Saving the model state before exiting')
        torch.save({ \
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(), \
                    'optimizer_state_dict': optimizer.state_dict(), \
                    'train_losses' : train_losses, \
                    'valid_losses' : valid_losses, \
                    'train_preds' : train_preds, \
                    'valid_preds' : valid_preds \
                    }, MODEL_SAVED \
                )


