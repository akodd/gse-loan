import torch as torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import tqdm
from os import path

from torch.utils.tensorboard import SummaryWriter

class FNMDatasetM2(Dataset):

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
        # we assume that idx_reduction keeps the same carditnality for cat. vars
        #'state_id', 'purpose_id', 'mi_type_id', 
        #'occupancy_status_id', 'product_type_id', 'property_type_id', 
        #'seller_id', 'zip3_id'
        self.acq_cat = np.max(acq[:, 1:], axis=0) # skip defaulted flag at 0 idx

    def __len__(self):
        return self.length
        #return 100
    
    def __getitem__(self, itemID):
        idx = self.idx_reduction[itemID]
        account = self.acq[idx, 1:] # 0 idx is default indicator
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

    def getAcqCat():
        return self.acq_cat


def paddingCollator(batch):
    seq_batch = [torch.from_numpy(batch[i][0]) for i in range(len(batch))]

    seq_lengths = [seq_batch[b].shape[0] for b in range(len(seq_batch))]
    seq_lengths, seq_perm_idx = torch.Tensor(seq_lengths).long().sort(0, descending=True)

    seq_batch = pad_sequence(seq_batch, batch_first=True)
    seq_batch = seq_batch[seq_perm_idx, :, :]

    acq_batch = [torch.from_numpy(batch[i][1]) for i in range(len(batch))]
    acq_batch = torch.stack(acq_batch)
    acq_batch = acq_batch[seq_perm_idx, :].long()

    default_1y_batch = [torch.from_numpy(batch[i][2]) for i in range(len(batch))]
    default_1y_batch = pad_sequence(default_1y_batch, batch_first=True)
    default_1y_batch = default_1y_batch[seq_perm_idx, :]

    return seq_batch, seq_lengths, acq_batch, default_1y_batch


class BaselM2Model(nn.Module):
    r"""
    Very simple model where we don't use any embedding or upb rescaling.
    We expect 4 features
    """
    def __init__(self, n_features, lstm_size, linear_size, embed_dims, embed_drp):
        """
        Parameters:
            n_features: number of features inside sequence. note that first
            8 (0+7) represent one hot encoding for DLQ
            lstm_size: the size of hidden unit in LSTM 
            linear_size: the number of of output units for the first linear
            embed_dims: the list of tuples, where the first element of tuple
                represents the number of categories and second element of the
                tuple if the embedding size
            embed_drp: drop out percentage after embedding layer
        """
        super(BaselM2Model, self).__init__()
        self.embed_layers = nn.ModuleList(\
            [nn.Embedding(c, d) for c, d in embed_dims])

        self.lstm = nn.LSTM(
            input_size= n_features,
            hidden_size=lstm_size,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )

        self.acq_bn_dim = sum(x[1] for x in embed_dims)
        self.lin_bn_dim = self.acq_bn_dim + linear_size

        self.embegging_dpout = nn.Dropout(embed_drp)
        self.embedding_bn = nn.BatchNorm1d(self.acq_bn_dim)
        self.linear = nn.Linear(lstm_size + self.acq_bn_dim, linear_size)
        self.linear_dpout = nn.Dropout(0.1)
        self.linear_bn = nn.BatchNorm1d(linear_size)
        self.linear_out = nn.Linear(linear_size, 1)


    def forward(self, seq_padded, acq, input_lengths, default_1y):
        # process sequence
        self.lstm.flatten_parameters()
        total_length = seq_padded.size(1)
        packed_input = pack_padded_sequence(seq_padded, input_lengths, 
                                            batch_first=True)
        packed_output, _  = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                total_length=total_length)
        
        # process embeddings
        #e = [self.embed_layers[var](acq[:, var]) for var in range(acq.shape[1])]
        # subtract one since there is no 0 index coming from Vertica
        e = [embed(acq[:,i]) for i, embed in enumerate(self.embed_layers)]
        e = torch.cat(e,  1)
        e = functional.relu(e)
        e = self.embedding_bn(e)
        e = self.embegging_dpout(e)

        default_1y_hat = torch.zeros_like(default_1y)
        for t in range(int(total_length)):
            seq_out = output[:, t, :]
            lin_in = torch.cat([e, seq_out], 1)
            lin_out = self.linear(lin_in)
            lin_out = functional.relu(lin_out)
            lin_out = self.linear_bn(lin_out)
            lin_out = self.linear_dpout(lin_out)
            default_1y_hat[:, t] = self.linear_out(lin_out).view(-1)
        default_1y_hat = torch.sigmoid(default_1y_hat)
        return default_1y_hat

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

def makeEmeddings(acq, embeding_dim=2):
    # add 1 since NULLs are extracted as 0
    acq_cat = np.max(acq[:, 1:], axis=0) + 1
    emb = [(acq_cat[i], embeding_dim) for i in range(acq_cat.shape[0])]
    return emb


if __name__ == "__main__":
    TRAIN_PATH = '/home/user/notebooks/data/train'
    VALID_PATH = '/home/user/notebooks/data/valid'
    MODEL_PATH = '/home/user/notebooks/data/model/baselm2_full'
    SUMMARY_PATH = '/home/user/notebooks/runs/baselm2/full'
    MODEL_SAVED = MODEL_PATH + '/baselm2.pth'

    BATCH_SIZE = 8000 # TODO: change back to 10000
    NUM_WORKERS = 5 # change back to 10 TODO:
    BATCH_STOP = 100
    NUM_EPOCHS = 100
    LOSS_EVERY_N_BATCHES=10
    SAVE_EVERY_N_BATCHES=100

    train_acq, train_idx_to_seq, train_seq = load_data(TRAIN_PATH)
    valid_acq, valid_idx_to_seq, valid_seq = load_data(VALID_PATH)

    acq_embs = makeEmeddings(train_acq, embeding_dim=2)
    
    train_ds = FNMDatasetM2(train_acq, train_idx_to_seq, train_seq, 0)
    valid_ds = FNMDatasetM2(valid_acq, valid_idx_to_seq, valid_seq, 0)

    print("Number of train acq: {:,}".format(len(train_ds)))
    print("Number of valid acq: {:,}".format(len(valid_ds)))
    
    trainDL = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle = True, \
        collate_fn=paddingCollator, num_workers=NUM_WORKERS)
    validDL = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle = True, \
        collate_fn=paddingCollator, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu') #TODO: change back to cuda

    model = BaselM2Model(n_features=11, lstm_size=75, linear_size=25, \
        embed_dims=acq_embs, embed_drp=0.1)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters())
    checkpoint_epoch = 0
    train_losses = []
    valid_losses = []

    writer = SummaryWriter(SUMMARY_PATH)
    dataiter = iter(trainDL)
    seq, seq_len, acq, def1y = dataiter.next()
    #seq = seq[0, :, :].view(1, -1)
    #writer.add_graph(model, (seq[0:1, :, :], seq_len[0:1], acq[0:1, :], def1y[0:1, :]), True)
    writer.add_graph(model, (seq, acq, seq_len, def1y), False)
    writer.flush()

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


    # TODO:
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
                    loss.backward()
                    optimizer.step()

                    if bidx % LOSS_EVERY_N_BATCHES  == 0:
                        mean_loss = np.mean(train_losses_epoch)
                        tqdm_inner.set_postfix(trainLoss = "{:.12f}".format(mean_loss))
                        writer.add_scalar('loss/training', mean_loss, training_iter)
                        writer.flush()

                    if bidx % SAVE_EVERY_N_BATCHES == SAVE_EVERY_N_BATCHES - 1:
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
                    for bidx, (seq, seq_len, acq, def1y) in enumerate(tqdm_inner):
                        tqdm_inner.set_description('Valid: %i' % bidx)
                        seq = seq.to(device)
                        acq = acq.to(device)
                        seq_len = seq_len.to(device)
                        def1y = def1y.to(device)
                        def1y_hat = model(seq, acq, seq_len, def1y)
                        loss = weightedLoss(def1y, def1y_hat, seq_len)
                        valid_losses_epoch.append(loss.item())

                        if bidx % LOSS_EVERY_N_BATCHES  == 0:
                            mean_loss = np.mean(valid_losses_epoch)
                            tqdm_inner.set_postfix( \
                                validLoss = "{:.12f}".format(mean_loss))
                            writer.add_scalar(\
                                    'loss/validation',
                                    mean_loss, validation_iter
                                )
                        validation_iter += 1

                t.set_postfix(\
                    trainLoss = "{:.12f}".format(np.mean(train_losses_epoch)), \
                    validLoss = "{:.12f}".format(np.mean(valid_losses_epoch)))

                # save losses and model
                torch.save({ \
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(), \
                    'optimizer_state_dict': optimizer.state_dict(), \
                    'train_losses' : train_losses, \
                    'valid_losses' : valid_losses
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
                    'valid_losses' : valid_losses
                    }, MODEL_SAVED \
                )
    writer.close()


