import torch as torch
import torch.nn as nn
import vertica_python
import pandas as pd
from torch.utils.data import *
import numpy as np
from os import path
import glob

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


        return sequence[['servicer_id', 'dlq', 'age', 'int_rate', 'current_upb', 'msa']], \
                account, np.array(sequence.default_1y)
        
if __name__ == "__main__":
    import os
    print(os.getcwd())
    a = FNMDataset(\
        acq_path = '/home/user/notebooks/data/fnm_input_acq_train.feather',
        seq_path = '/home/user/notebooks/data/fnm_input_seq_train_0.feather'
    )
    print(a.__len__())
    seq, acq, default_1y = a.__getitem__(64653)
    seq, acq, default_1y = a.__getitem__(64654)
    seq, acq, default_1y = a.__getitem__(64655)

    import time
    start = time.time()
    for idx in range(1, 100):
        seq, acq, default_1y = a.__getitem__(idx)
    end = time.time()
    print("Time {:,} seconds".format(end-start))
    print('Time per item {} seconds'.format((end-start)/100.0))