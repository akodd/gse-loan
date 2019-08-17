# sequence to 12 model

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
from os import path

DEBUG = False
QA_CHECK = True

bad_idx = -1;
bad_chunk = -1;
bad_id_in_chunk = -1;

class FNMDatasetS12(Dataset):
    """
        Constructs numpy based time series dataset.
        Parameters:
            acq: one row per account with the following fields:
                acq_def_ind, state_id, purpose_id, mi_type_id, 
                occupancy_status_id, product_type_id, property_type_id, 
                seller_id, zip3_id. Each variable starts with 1 and 0 is
                reserved to NULL values
            idx_to_seq: index matched index in acq table. Each row corresponds
                to an account. First column maps to the index in seq list to
                find correponding chunk. Second and third map to begin and end
                index in the seq chunk
            seq: the list of numpy arrays. Each array has the following fiels:
                default_1y, yyyymm, dlq_adj, age, int_rate, current_upb_norm
            ratio: if above zero then non defaulted are scaled as ratio of defaulted
                else no subsampling
    """
    def __init__(self, acq, idx_to_seq, seq, ratio = 0):
        self.predict_ahead = 12 # hardcode for now
        self.acq = acq
        self.seq = seq
        self.idx_to_seq = idx_to_seq
        ## clean up short sequencues
        good_seq_idx = np.where(idx_to_seq[:, 2] - idx_to_seq[:, 1] >= 24)
        self.idx_to_seq = self.idx_to_seq[good_seq_idx]
        self.acq = self.acq[good_seq_idx]
        # end clean up
        if ratio > 0:
            defaulted = self.acq[:,0]==1
            n_defaulted = np.sum(defaulted)
            def_idx = np.random.choice(np.where(defaulted)[0], n_defaulted, False)
            nondef_idx = np.random.choice(np.where(~defaulted)[0], int(n_defaulted*ratio) , False)
            sub_idx = np.concatenate([def_idx, nondef_idx])
            self.length = sub_idx.shape[0]
            self.idx_reduction = sub_idx
        else:
            self.length = self.acq.shape[0]
            self.idx_reduction = np.arange(self.length)
        self.acq_cat = np.max(self.acq[:, 1:], axis=0) # skip defaulted flag at 0 idx

    def __len__(self):
        return self.length
        #return 100
    
    def __getitem__(self, itemID):
        bad_idx = itemID
        idx = self.idx_reduction[itemID]
        account = self.acq[idx, 1:] # 0 idx is default indicator
        seq_info = self.idx_to_seq[idx, :]
        chunk_num = seq_info[0]
        idx_begin = seq_info[1]
        idx_end = seq_info[2]

        bad_chunk = seq_info
        bad_id_in_chunk = chunk_num
        

        # skip default_1y, yyyymm
        if DEBUG:
            sequence = self.seq[0][idx_begin:idx_end, 2:]
        else:
            sequence = self.seq[chunk_num][idx_begin:idx_end, 2:]
        dlq = sequence[:, 2].astype(int)
        dlq_one_hot = np.eye(19, dtype=np.float32)[dlq[:-self.predict_ahead]] # dlq is between 0 and 6 + 12
        sequence = np.concatenate([
            dlq_one_hot, 
            sequence[:-self.predict_ahead, 1:]], axis=1)

        target_dlq = dlq[-self.predict_ahead:]
        #target_upb = sequence[-self.predict_ahead:, 5]
        #target = np.concatenate([target_dlq, target_upb], axis=1)
        # if QA_CHECK:
        #     if (target_dlq.shape[0]!=12):
        #         print('\n####\n')
        #         print('chunk_iD: {}'.format(chunk_num))
        #         print('itemID' + str(itemID))
        #         print('begin: end: {idx_begin}')
        #         print('begin: end: {}:{}'.format(idx_begin, idx_end))

        return sequence, account, target_dlq # predict DLQ for now

    def getAcqCat():
        return self.acq_cat

class BaselS12Model(nn.Module):
    r"""
     Fill up
    """
    def __init__(self, seq_n_features, lstm_size, lstm_layers, lstm_dropout, 
                    embed_dims):
        """
        Parameters:
            seq_n_features: number of features inside sequence. note that first
            8 (0+7) represent one hot encoding for DLQ
            lstm_size: the size of hidden unit in LSTM 
            linear_size: the number of of output units for the first linear
            embed_dims: the list of tuples, where the first element of tuple
                represents the number of categories and second element of the
                tuple if the embedding size
            embed_drp: drop out percentage after embedding layer
        """
        super(BaselS12Model, self).__init__()
        self.embed_layers = nn.ModuleList(\
            [nn.Embedding(c, d) for c, d in embed_dims])

        # total dim from embeddings
        self.acq_bn_dim = sum(x[1] for x in embed_dims)
        self.embedding_bn = nn.BatchNorm1d(self.acq_bn_dim)
        lstm_input_dim = seq_n_features + self.acq_bn_dim
        self.lstm = nn.LSTM(
            input_size  = lstm_input_dim,
            hidden_size = lstm_size,
            num_layers  = lstm_layers,
            dropout     = lstm_dropout,
            batch_first  = True
        )

        self.linear1 = nn.Linear(lstm_size, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.dpout1 = nn.Dropout(0.5)

        self.linear2 = nn.Linear(100, 250)
        self.bn2 = nn.BatchNorm1d(250)
        self.dpout2 = nn.Dropout(0.5)
        
        self.linear3 = nn.Linear(250, 19*12)
        self.bn3 = nn.BatchNorm1d(19*12)
        #self.dpout1 = nn.Dropout(0.1)

        #self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, seq_padded, input_lengths, acq):
        # embed acq
        acq = acq.long()
        e = [embed(acq[:,i]) for i, embed in enumerate(self.embed_layers)]
        e = torch.cat(e,  1)
        e = functional.relu(e)
        e = self.embedding_bn(e)
        # reshape and replicate e to attach to each time index
        e = e.reshape(seq_padded.shape[0], -1, e.shape[1])\
            .expand(-1, seq_padded.shape[1], -1)
        # glue embeddings to each time index of the seq
        s = torch.cat([seq_padded, e], 2)

        # process sequence
        self.lstm.flatten_parameters()
        #total_length = seq_padded.size(1)
        packed_input = pack_padded_sequence(s, input_lengths, 
                                            batch_first=True)
        packed_output, (ht, _)  = self.lstm(packed_input)

        # ht is the final element of the sequence shape (hidden_size, 1)
        # we are predicting 19 DLQ x 12 months
        out = ht[-1, :, :]
        out = self.bn1(functional.relu(self.linear1(out)))
        out = self.dpout1(out)

        out = self.bn2(functional.relu(self.linear2(out)))
        out = self.dpout2(out)

        out = self.bn3(functional.relu(self.linear3(out)))
        #out = self.dpout2(out)

        out = out.reshape(-1, 19, 12) 
        #out = self.logsoftmax(out)
        return out


def paddingCollator(batch):
    seq_batch = [torch.from_numpy(batch[i][0]) for i in range(len(batch))]

    seq_lengths = [seq_batch[b].shape[0] for b in range(len(seq_batch))]
    seq_lengths, seq_perm_idx = torch.Tensor(seq_lengths).int().sort(0, descending=True)

    seq_batch = pad_sequence(seq_batch, batch_first=True)
    seq_batch = seq_batch[seq_perm_idx, :, :]

    acq_batch = [torch.from_numpy(batch[i][1]) for i in range(len(batch))]
    acq_batch = torch.stack(acq_batch)
    acq_batch = acq_batch[seq_perm_idx, :]

    target_batch = [torch.from_numpy(batch[i][2]) for i in range(len(batch))]
    target_batch = torch.stack(target_batch)
    target_batch = target_batch[seq_perm_idx, :]    

    return seq_batch, seq_lengths, acq_batch, target_batch

def load_data(data_path):
    acquisition_nname = data_path + '/fnm_input_acq.npy'
    if DEBUG:
        sequence_nname = data_path + '/fnm_input_seq_0.npy' 
    else:
        sequence_nname = data_path + '/fnm_input_seq_*.npy' 
    idx_to_seq_nname = data_path + '/fnm_input_idx_to_seq.npy'
    
    print('Acquisition numpy: {}'.format(acquisition_nname))
    print('Sequence numpy: {}'.format(sequence_nname))
    print('Index to Sequence Index numpy: {}'.format(idx_to_seq_nname))
    
    seq_files = sorted([f for f in glob.glob(sequence_nname, recursive=False)])
    seq_numpy = [None] * len(seq_files)
    
    acq_numpy = np.load(acquisition_nname)
    idx_to_seq = np.load(idx_to_seq_nname)
    
    print("loading seq_numpy_chunk")
    for chunk_idx, seq_numpy_chunk in enumerate(tqdm.tqdm(seq_files)):
        seq_numpy[chunk_idx] = np.load(seq_numpy_chunk)
        
    return acq_numpy, idx_to_seq, seq_numpy

def makeEmeddings(acq, embeding_dim=2):
    # add 1 since NULLs are extracted as 0
    acq_cat = np.max(acq[:, 1:], axis=0) + 1
    emb = [(acq_cat[i], embeding_dim) for i in range(acq_cat.shape[0])]
    return emb



if __name__ == "__main__":
    TRAIN_PATH = '/home/user/notebooks/data/train'
    VALID_PATH = '/home/user/notebooks/data/valid'
    MODEL_PATH = '/home/user/notebooks/data/model/baselS12'
    SUMMARY_PATH = '/home/user/notebooks/runs/baselS12'
    MODEL_SAVED = MODEL_PATH + '/baselS12.pth'

    if DEBUG:
        BATCH_SIZE = 4
        NUM_WORKERS = 0 
    else:
        BATCH_SIZE = 512 #12*2**10
        NUM_WORKERS = 16
    #BATCH_STOP = 100
    NUM_EPOCHS = 10
    LOSS_EVERY_N_BATCHES=200
    SAVE_EVERY_N_BATCHES=1000

    train_acq, train_idx_to_seq, train_seq = load_data(TRAIN_PATH)
    valid_acq, valid_idx_to_seq, valid_seq = load_data(VALID_PATH)

    train_ds = FNMDatasetS12(train_acq, train_idx_to_seq, train_seq, 8)
    valid_ds = FNMDatasetS12(valid_acq, valid_idx_to_seq, valid_seq, 8)

    print("Number of train acq: {:,}".format(len(train_ds)))
    print("Number of valid acq: {:,}".format(len(valid_ds)))
    
    trainDL = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle = True, \
        collate_fn=paddingCollator, num_workers=NUM_WORKERS)
    validDL = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle = True, \
        collate_fn=paddingCollator, num_workers=NUM_WORKERS)

    if DEBUG:
        device = torch.device('cpu')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    acq_embs = makeEmeddings(train_acq, embeding_dim=2)

    model = BaselS12Model(\
        seq_n_features=22, lstm_size=100, lstm_layers=2, lstm_dropout=0.5, 
            embed_dims = acq_embs)
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(params=model.parameters())
    checkpoint_epoch = 0
    train_losses = []
    valid_losses = []

    writer = SummaryWriter(SUMMARY_PATH)
   #  dataiter = iter(trainDL)
   #  seq, seq_len, acq, target_dlq = dataiter.next()
   #  writer.add_graph(model, (seq, seq_len, acq), False)
   #  writer.flush()
    
    model = model.to(device)
    loss_function = loss_function.to(device)

    if path.exists(MODEL_SAVED):
        print('Loading model checkpoint: {}'.format(MODEL_SAVED))
        checkpoint = torch.load(MODEL_SAVED)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        valid_losses = checkpoint['valid_losses']

    if not DEBUG:
        if torch.cuda.device_count() > 1:
            print("Training on", torch.cuda.device_count(), "GPUs")
            model = nn.DataParallel(model)

    training_iter = 0
    validation_iter = 0

    try:
        with tqdm.trange(checkpoint_epoch, checkpoint_epoch + NUM_EPOCHS) as t:
            for epoch in t:
                train_losses_epoch = []
                t.set_description('Epoch: %i' % epoch)
                model.train()
                tqdm_inner = tqdm.tqdm(trainDL)
                for bidx, (seq, seq_len, acq, target) in enumerate(tqdm_inner):
                    tqdm_inner.set_description('Train: %i' % bidx)
                    model.zero_grad()
                    seq = seq.to(device)
                    acq = acq.to(device)
                    seq_len = seq_len.to(device)
                    target = target.to(device)

                    target_hat = model(seq, seq_len, acq)
                    loss = loss_function(target_hat, target)
                    train_losses_epoch.append(loss.item())
                    loss.backward()
                    optimizer.step()

                    if bidx % LOSS_EVERY_N_BATCHES  == 0:
                        mean_loss = np.sum(train_losses_epoch)/len(train_losses_epoch)/BATCH_SIZE
                        tqdm_inner.set_postfix(trainLoss = "{:.12f}".format(mean_loss))
                        writer.add_scalar('loss/training', mean_loss, training_iter)
                        writer.flush()

                    if bidx % SAVE_EVERY_N_BATCHES == SAVE_EVERY_N_BATCHES - 1:
                        if DEBUG:
                            torch.save({ \
                                    'epoch': epoch,
                                    'model_state_dict': model.state_dict(), \
                                    'optimizer_state_dict': optimizer.state_dict(), \
                                    'train_losses' : train_losses, \
                                    'valid_losses' : valid_losses \
                                    }, MODEL_SAVED \
                                )
                        else:
                            torch.save({ \
                                'epoch': epoch,
                                'model_state_dict': model.module.state_dict(), \
                                'optimizer_state_dict': optimizer.state_dict(), \
                                'train_losses' : train_losses, \
                                'valid_losses' : valid_losses \
                                }, MODEL_SAVED \
                            )
                    training_iter += 1

                model.eval()
                with torch.no_grad():
                    valid_losses_epoch = []
                    tqdm_inner = tqdm.tqdm(validDL)
                    for bidx, (seq, seq_len, acq, target) in enumerate(tqdm_inner):
                        tqdm_inner.set_description('Valid: %i' % bidx)
                        seq = seq.to(device)
                        acq = acq.to(device)
                        seq_len = seq_len.to(device)
                        target = target.to(device)
                        target_hat = model(seq, seq_len, acq)
                        loss = loss_function(target_hat, target)
                        valid_losses_epoch.append(loss.item())

                        if bidx % LOSS_EVERY_N_BATCHES  == 0:
                            mean_loss = np.sum(valid_losses_epoch)/len(valid_losses_epoch)/BATCH_SIZE
                            tqdm_inner.set_postfix( \
                                validLoss = "{:.12f}".format(mean_loss))
                            writer.add_scalar(\
                                    'loss/validation',
                                    mean_loss, validation_iter
                                )
                        validation_iter += 1

                t.set_postfix(\
                    trainLoss = "{:.12f}".format(np.sum(train_losses_epoch)/len(train_losses_epoch)/BATCH_SIZE), \
                    validLoss = "{:.12f}".format(np.sum(valid_losses_epoch)/len(valid_losses_epoch)/BATCH_SIZE))

                # save losses and model
                if DEBUG:
                    torch.save({ \
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(), \
                            'optimizer_state_dict': optimizer.state_dict(), \
                            'train_losses' : train_losses, \
                            'valid_losses' : valid_losses \
                            }, MODEL_SAVED \
                        )
                else:
                    torch.save({ \
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(), \
                        'optimizer_state_dict': optimizer.state_dict(), \
                        'train_losses' : train_losses, \
                        'valid_losses' : valid_losses \
                        }, MODEL_SAVED \
                    )

                train_losses.append(np.sum(train_losses_epoch)/len(train_losses_epoch)/BATCH_SIZE)
                valid_losses.append(np.sum(valid_losses_epoch)/len(valid_losses_epoch)/BATCH_SIZE)

    except KeyboardInterrupt:
        print ('Saving the model state before exiting')
        if DEBUG:
            torch.save({ \
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(), \
                    'optimizer_state_dict': optimizer.state_dict(), \
                    'train_losses' : train_losses, \
                    'valid_losses' : valid_losses \
                    }, MODEL_SAVED \
                )
        else:
            torch.save({ \
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(), \
                'optimizer_state_dict': optimizer.state_dict(), \
                'train_losses' : train_losses, \
                'valid_losses' : valid_losses \
                }, MODEL_SAVED \
            )
    except RuntimeError as e:
        print(e)
        print("####### bad: {}, {}, {}".format(bad_idx, bad_chunk, bad_id_in_chunk))
    writer.close()
