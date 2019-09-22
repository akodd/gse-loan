import torch as torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import numpy as np
import tqdm


from dsetutil import load_data


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
            ym2idx: dictionary of yyyymm integer to index in macros mapping
            ratio: if above zero then non defaulted are scaled as ratio of defaulted
                else no subsampling
    """
    def __init__(self, acq, idx_to_seq, seq, ym2idx, dlq_dim, ratio = 0):
        self.predict_ahead = 12
        self.dlq_dim = dlq_dim
        self.acq = acq
        self.seq = seq
        self.idx_to_seq = idx_to_seq
        self.ym2idx = ym2idx
        ## clean up short sequencues
        # give at least 12 months to train - hardcode for now
        print('Initial acq: {:,}'.format(self.acq.shape[0]))
        good_seq_idx = np.where(idx_to_seq[:, 2] - idx_to_seq[:, 1] >= 12 + self.predict_ahead)
        self.idx_to_seq = self.idx_to_seq[good_seq_idx]
        self.acq = self.acq[good_seq_idx]
        # end clean up
        print('Non-short acq: {:,}'.format(self.acq.shape[0]))
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
    
    def __getitem__(self, itemID):
        idx = self.idx_reduction[itemID]
        account = self.acq[idx, 1:] # 0 idx is default indicator
        seq_info = self.idx_to_seq[idx, :]
        chunk_num = seq_info[0]
        idx_begin = seq_info[1]
        idx_end = seq_info[2]

        # skip default_1y
        sequence = self.seq[chunk_num][idx_begin:idx_end, 1:]

        dlq = sequence[:, 1].astype(int)
        np.clip(dlq, 0, self.dlq_dim-1, out=dlq)
        dlq_one_hot = np.eye(self.dlq_dim, dtype=np.float32)[dlq[:-self.predict_ahead]] # dlq is between 0 and 6 + 12
        
        ymb, yme = int(self.ym2idx[sequence[0, 0]]), int(self.ym2idx[sequence[-1, 0]])
        yymmsamod = np.concatenate([
            np.arange(ymb, yme+1-self.predict_ahead).reshape(-1, 1),
            sequence[:-self.predict_ahead, [-2, -1]]
        ], axis=1).astype(np.int64)
        
        sequence = np.concatenate([
            sequence[:-self.predict_ahead, 1:], # yyyymm is column 0
            dlq_one_hot], axis=1)

        account = account.astype(np.int64)
        target_dlq = dlq[-self.predict_ahead:]

        return sequence, yymmsamod, account, target_dlq

def paddingCollator(batch):
    # sequence, account, yyyymm, target_dlq
    seq_batch = [torch.from_numpy(batch[i][0]) for i in range(len(batch))]

    seq_lengths = [seq_batch[b].shape[0] for b in range(len(seq_batch))]
    seq_lengths, seq_perm_idx = torch.Tensor(seq_lengths).int().sort(0, descending=True)

    seq_batch = pad_sequence(seq_batch, batch_first=True)
    seq_batch = seq_batch[seq_perm_idx, :, :]

    ymd_batch = [torch.from_numpy(batch[i][1]) for i in range(len(batch))]
    ymd_batch = pad_sequence(ymd_batch, batch_first=True)
    ymd_batch = ymd_batch[seq_perm_idx, :, :]

    acq_batch = [torch.from_numpy(batch[i][2]) for i in range(len(batch))]
    acq_batch = torch.stack(acq_batch)
    acq_batch = acq_batch[seq_perm_idx, :]

    target_batch = [torch.from_numpy(batch[i][3]) for i in range(len(batch))]
    target_batch = torch.stack(target_batch)
    target_batch = target_batch[seq_perm_idx, :]

    return seq_batch, seq_lengths, ymd_batch, acq_batch, target_batch


if __name__ == "__main__":
    TRAIN_PATH = '/home/user/notebooks/data/train'
    VALID_PATH = '/home/user/notebooks/data/valid'
    TEST_PATH  = '/home/user/notebooks/data/test'

    acq, idx_to_seq, seq, macro, ym2idx = load_data(TRAIN_PATH, True, True)
    dataset = FNMDatasetS12(acq, idx_to_seq, seq, ym2idx, dlq_dim=19)


    print('Sequencies: {:,}'.format(len(dataset)))

    BATCH_SIZE = 512
    NUM_WORKERS = 0

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle = True, \
        collate_fn=paddingCollator, num_workers=NUM_WORKERS)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for bidx, (seq, seq_len, ymd, acq, target) in enumerate(tqdm.tqdm(data_loader)): 
        seq = seq.to(device)
        ymd = seq.to(ymd)
        acq = acq.to(device)
        seq_len = seq_len.to(device)
        target = target.to(device)