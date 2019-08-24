# FNM with Macros dataset

import torch as torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import tqdm
import pickle
import time

class FNMMacrosDataset(Dataset):
    """
        Constructs numpy based time series dataset with macro economic
        variables
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
            macros: the numpy array of macro economic varaibles
            ym2idx: dictionary of yyyymm integer to index in macros mapping
            ratio: if above zero then non defaulted are scaled as ratio of defaulted
                else no subsampling
    """
    def __init__(self, acq, idx_to_seq, seq, macros, ym2idx, prediction_length=12, ratio = 0):
        self.predict_ahead = prediction_length
        self.acq = acq
        self.seq = seq
        self.idx_to_seq = idx_to_seq
        self.macros = macros
        self.ym2idx = ym2idx
        ## clean up short sequencues
        # give at least 12 months to train - hardcode for now
        print('Initial acq: {:,}'.format(self.acq.shape[0]))
        good_seq_idx = np.where(idx_to_seq[:, 2] - idx_to_seq[:, 1] >= 12 + self.predict_ahead)
        self.idx_to_seq = self.idx_to_seq[good_seq_idx]
        self.acq = self.acq[good_seq_idx]
        # end clean up
        print('Non-short acq: {:,}'.format(self.acq.shape[0]))
        # remove accounts with gaps
        #gap_idx = np.array(self.gap(), dtype=np.int)
        #self.idx_to_seq = self.idx_to_seq[-gap_idx]
        #self.acq = self.acq[-gap_idx]
        # end gapless
        #print('Gapless acq: {:,}'.format(self.acq.shape[0]))
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

    def gap(self):
        gap_idx = []
        for idx in tqdm.trange(self.idx_to_seq.shape[0]):
            seq_info = self.idx_to_seq[idx, :]
            chunk_num = seq_info[0]
            idx_begin = seq_info[1]
            idx_end = seq_info[2]
            ym = self.seq[chunk_num][idx_begin:idx_end, 1]
            if not np.all((ym == 1) + (ym == 89)):
                gap_idx.append(idx)
        return gap_idx
        
    
    def __getitem__(self, itemID):
        idx = self.idx_reduction[itemID]
        account = self.acq[idx, 1:] # 0 idx is default indicator
        seq_info = self.idx_to_seq[idx, :]
        chunk_num = seq_info[0]
        idx_begin = seq_info[1]
        idx_end = seq_info[2]

        # skip default_1y
        sequence = self.seq[chunk_num][idx_begin:idx_end, 1:]
        dlq = sequence[:, 1].astype(int) # yyyymm, dlq
        dlq_one_hot = np.eye(19, dtype=np.float32)[dlq[:-self.predict_ahead]] # dlq is between 0 and 6 + 12

        ymb, yme = int(self.ym2idx[sequence[0, 0]]), int(self.ym2idx[sequence[-1, 0]])
        macro = self.macros[ymb:(yme+1)]

        yyyymm = sequence[:-self.predict_ahead, 0]

        # print(itemID, idx, seq_info, sequence[:-self.predict_ahead, 1:].shape, 
        #     macro[:-self.predict_ahead].shape,
        #     dlq_one_hot.shape)

        macro_out = macro[:-self.predict_ahead]
        seq_out = sequence[:-self.predict_ahead, 1:]
        if (macro_out.shape[0]!=seq_out.shape[0]):
            print('Macro gap: ' + seq_info)

        sequence = np.concatenate([
            sequence[:-self.predict_ahead, 1:], # yyyymm is column 0
            macro[:-self.predict_ahead], 
            dlq_one_hot], axis=1)

        target_dlq = dlq[-self.predict_ahead:]
        #target_upb = sequence[-self.predict_ahead:, 5]
        #target = np.concatenate([target_dlq, target_upb], axis=1)

        return sequence, account, target_dlq # predict DLQ for now

    def getAcqCat():
        return self.acq_cat


def paddingCollator(batch):
    # sequence, account, yyyymm, target_dlq
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


def load_data(data_path, verbose=True):
    acquisition_nname = data_path + '/fnm_input_acq.npy'
    sequence_nname = data_path + '/fnm_input_seq_*.npy' 
    idx_to_seq_nname = data_path + '/fnm_input_idx_to_seq.npy'
    macros_nname = data_path + '/fred_norm.npy'
    ym2idx_nname = data_path + '/ym2idx.pickle'
    
    if verbose:
        print('Acquisition: {}'.format(acquisition_nname))
        print('Sequence: {}'.format(sequence_nname))
        print('Index to Sequence Index: {}'.format(idx_to_seq_nname))
        print('Macros: {}'.format(macros_nname))
        print('YYYYMM to Index: {}'.format(ym2idx_nname))
 
    acq = np.load(acquisition_nname)
    idx_to_seq = np.load(idx_to_seq_nname)
    macros = np.load(macros_nname)
    #ym2idx = np.load(ym2idx_nname)
    with open(ym2idx_nname, 'rb') as f:
        ym2idx = pickle.load(f)

    seq_files = sorted([f for f in glob.glob(sequence_nname, recursive=False)])
    seq = [None] * len(seq_files)
    
    if verbose:
        print("loading seq_chunk")
        itt = tqdm.tqdm(seq_files)
    else:
        itt = seq_files
    for chunk_idx, seq_chunk in enumerate(itt):
        if verbose:
            itt.set_description('Loading: ' + seq_chunk)
        seq[chunk_idx] = np.load(seq_chunk)
        
    return acq, idx_to_seq, seq, macros, ym2idx


if __name__ == "__main__":
    TRAIN_PATH = '/home/user/notebooks/data/train'
    VALID_PATH = '/home/user/notebooks/data/valid'
    TEST_PATH  = '/home/user/notebooks/data/test'

    def load_report(data_path):
        acq, idx_to_seq, seq, macro, ym2idx = load_data(data_path, True)
        print('acq.shape={}'.format(acq.shape))
        print('idx_to_seq.shape={}'.format(idx_to_seq.shape))
        print('macro.shape={}'.format(macro.shape))
        print('len(ym2idx)={}'.format(len(ym2idx)))

        for i in range(len(seq)):
            print('\tseq[{}].shape={}'.format(i, seq[i].shape))


    # print('TRAIN DATA #######')
    # load_report(TRAIN_PATH)
    # print('VALID DATA #######')
    # load_report(VALID_PATH)
    # print('TEST DATA #######')
    # load_report(TEST_PATH)

    acq, idx_to_seq, seq, macro, ym2idx = load_data(TRAIN_PATH)
    dataset = FNMMacrosDataset(acq, idx_to_seq, seq, macro, ym2idx)

    print('Sequencies: {:,}'.format(len(dataset)))

    BATCH_SIZE = 256
    NUM_WORKERS = 10

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle = True, \
        collate_fn=paddingCollator, num_workers=NUM_WORKERS)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')

    for bidx, (seq, seq_len, acq, target) in enumerate(tqdm.tqdm(data_loader)): 
        seq = seq.to(device)
        acq = acq.to(device)
        seq_len = seq_len.to(device)
        target = target.to(device)
