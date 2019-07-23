import torch as torch
import torch.nn as nn
import vertica_python
import pandas as pd
from torch.utils.data import *
import numpy as np
from os import path

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

    def __init__(self, acq_path, seq_paths,  number_of_loans = 1000):
        self.fnm_encode_acq = pd.read_feather(acq_path)\
            .set_index('loan_id')
        self.length = len(self.fnm_encode_acq.index)
        self.seq_paths = seq_paths
        print("Number of loans: {}".format(self.length))
        for _, seq_path in enumerate(seq_paths):
            if not path.exists(seq_path):
                raise ValueError("Cannot find feather file.", "Path: {}".format(seq_path))

    def getIDs(self, seq_path_idx):
        self.fnm_input_seq = pd.read_feather(self.seq_paths[seq_path_idx])
        self.loan_ids = self.fnm_input_seq.loan_id.unique()
        return self.loan_ids

    def __len__(self):
        #return self.length
        return self.length
    

    def __getitem__(self, idx):
        # self.loan_ids contains ids that are indexed by idx
        # extract row from acq by loan_id
        # extract from self.fnm_input_seq by loan_id ordered by rptperiod
        # merge them

        loan_id = self.fnm_encode_acq.loc[idx, :]
        cur = self.connection.cursor()
        cur.execute(select_seq_sql.format(loan_id))
        res = cur.fetchall()
        res_pd = pd.DataFrame(res, columns=seq_col_name).astype({'loan_id':'int64'})
        loan_pd = pd.DataFrame(data=[self.fnm_encode_acq.loc[idx, :].values], \
            columns=self.fnm_encode_acq.columns)
        res_pd = res_pd.merge(loan_pd)

        
        # extracting default indicator
        
        # extracting sequence 

        return res_pd, res_pd.default_1y
        
if __name__ == "__main__":
    import os
    print(os.getcwd())

    a = FNMDataset('notebooks/data/fnm_encode_acq_train.feather', 0)
    print(a.__len__())
    b, _ = a.__getitem__(64653)
    print(b)
    b = a.__getitem__(64654)
    print(b)
    b = a.__getitem__(64655)
    print(b)

    import time
    start = time.time()
    for idx in range(1, 10000):
        b, _ = a.__getitem__(idx)
    end = time.time()
    print('Time {} seconds'.format(end-start))