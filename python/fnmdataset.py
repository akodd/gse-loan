import torch as torch
import torch.nn as nn
import vertica_python
import pandas as pd
from torch.utils.data import *
import numpy as np
from os import path
import glob
from torch.nn.utils.rnn import *

conn_info = {
    'host': 'vertica',
    'port': 5433,
    'user': 'dbadmin',
    'session_label': 'fnm',
    'unicode_error': 'strict',
    'ssl': False,
    'use_prepared_statements': True,
    'connection_timeout': 120
}

select_seq_sql = """
    select
        loan_id,
        rpt_period,
        servicer_id,
        default_1y,
        dlq,
        age,
        int_rate,
        current_upb,
        months_to_maturity,
        msa,
        modification_flag
    from fnm_input_ds
    where loan_id = '{}'
    order by rpt_period
"""

seq_col_name =  [
    'loan_id',
    'rpt_period',
    'servicer_id',
    'default_1y',
    'dlq',
    'age',
    'int_rate',
    'current_upb',
    'months_to_maturity',
    'msa',
    'modification_flag'
]

class FNMRandomBatchSampler(BatchSampler):
    r"""
    
    """

    def __init__(self, data_source, files, batch_size):
        self.data_source = data_source
        self.mega_batch_number = len(files)
        self.current_mega_batch = 0
        self.batch_size = batch_size
        
    def __iter__(self):
       for mbatch_idx in range(self.mega_batch_number):
           mbatch_ids = self.data_source.getIDs(mbatch_idx)
           np.shuffle(mbatch_ids)
           batch = []
           for idx in mbatch_ids:
               batch.append(idx)
               if len(batch) == self.batch_size:
                   yield batch
                   batch = []
            # always drop last for now

    def __len__(self):
        return len(self.data_source) // self.batch_size

class FNMDataset(Dataset):

    def __init__(self, acq_path, seq_path,  number_of_loans = 1000):
        # load acq files
        self.acq = pd.read_feather(acq_path).set_index('loan_id')
        self.length = len(self.acq.index)
        # load seq files
        self.seq_path = seq_path
        files = [f for f in glob.glob(seq_path, recursive=False)]
        if len(files)<=0:
            raise ValueError('Cannot find seq files', "Path: {}".format(seq_path))
        self.seq_fnames = { int(files[i].split('/')[-1].split('.')[0].split('_')[-1]):files[i] for i in range(len(files))}
        self.seq_feathers = { idx: pd.read_feather(fname).set_index(['loan_id']) \
            for idx, fname in self.seq_fnames.items()}

        # TODO: create mapping between r and seq feather file
        # override the number of loans for now
        self.length = len(self.seq_feathers[0].index.unique())
        self.acq = self.acq[self.acq.r < 0.08]

    #def getIDs(self, seq_path_idx):
    #    self.fnm_input_seq = pd.read_feather(self.seq_paths[seq_path_idx])
    #    self.loan_ids = self.fnm_input_seq.loan_id.unique()
    #    return self.loan_ids

    def __len__(self):
        #return self.length
        return self.length
    

    def __getitem__(self, idx):
        # self.loan_ids contains ids that are indexed by idx
        # extract row from acq by loan_id
        # extract from self.fnm_input_seq by loan_id ordered by rptperiod
        # merge them
        # TODO: extend to many seq feather files

        account = self.acq.iloc[[idx], :]
        sequence = self.seq_feathers[0].loc[account.index[0], :]
        sequence = sequence.sort_values('yyyymm')\
            .fillna(method='bfill').fillna(method='ffill')

        return \
            sequence[['dlq', 'age', 'int_rate', 'current_upb']].to_numpy(), \
            account[['state_id', 'purpose_id', 'mi_type_id', \
                'occupancy_status_id', 'product_type_id', 'property_type_id', \
                    'seller_id', 'zip3_id']].fillna(0).to_numpy(), \
            np.array(sequence.default_1y)
        
if __name__ == "__main__":
    import os
    print(os.getcwd())
    train_ds = FNMDataset(\
        acq_path = '/home/user/notebooks/data/fnm_input_acq_train.feather',
        seq_path = '/home/user/notebooks/data/fnm_input_seq_train_0.feather'
    )
    print(len(train_ds))
    seq, acq, default_1y = train_ds.__getitem__(64653)
    seq, acq, default_1y = train_ds.__getitem__(64654)
    seq, acq, default_1y = train_ds.__getitem__(64655)

    def my_pad_sequence(batch):
        seq_batch = [torch.from_numpy(batch[i][0]) for i in range(len(batch))]
        seq_batch = pad_sequence(seq_batch, batch_first=True)

        acq_batch = [torch.from_numpy(batch[i][1]) for i in range(len(batch))]
        acq_batch = torch.stack(acq_batch)

        default_1y_batch = [torch.from_numpy(batch[i][2]) for i in range(len(batch))]
        default_1y_batch = pad_sequence(default_1y_batch, batch_first=True)
        return seq_batch, acq_batch, default_1y_batch

    dataLoader = DataLoader(train_ds, batch_size=7,
        collate_fn=my_pad_sequence)
    for batch_idx, (sequence, account, default_1y) in enumerate(dataLoader):
        print('batch_idx: {}'.format(batch_idx))
        print('sequence shape: {}'.format(sequence.shape))
        print('account shape: {}'.format(account.shape))
        print('default_1y shape: {}'.format(default_1y.shape))

    import time
    start = time.time()
    for idx in range(1, 100):
        seq, acq, default_1y = a.__getitem__(idx)
    end = time.time()
    print("Time {:,} seconds".format(end-start))
    print('Time per item {} seconds'.format((end-start)/100.0))