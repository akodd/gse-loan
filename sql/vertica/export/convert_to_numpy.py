#!/usr/bin/python

import pandas as pd
import numpy as np
import glob
import gc
import cudf
import os

def convertAcq(acq_filename, includeLoanID = False):
    acq = pd.read_parquet(acq_filename).sort_values('loan_id').reset_index(drop=True)
    acq = acq.fillna(0)
    if includeLoanID:
        acq_numpy = acq[['loan_id', 'acq_def_ind', 'state_id', 'purpose_id', 'mi_type_id', \
                    'occupancy_status_id', 'product_type_id', 'property_type_id', \
                        'seller_id', 'zip3_id']].to_numpy()
    else:
        acq_numpy = acq[[           'acq_def_ind', 'state_id', 'purpose_id', 'mi_type_id', \
            'occupancy_status_id', 'product_type_id', 'property_type_id', \
                'seller_id', 'zip3_id']].to_numpy(dtype=np.int32)
    return acq_numpy

def convertSeq(files, includeLoanID = False):
    # mapping of index or loan id to chunk and sequence offsets
    lid_to_seq_idx =  []     
    seq_numpy = [None] * len(files)
    #seq_fnames = { int(files[i].split('/')[-1].split('.')[0].split('_')[-1]): files[i] for i in range(len(files))}

    for i, fname in enumerate(files):
        print('processing name: {}'.format(fname))
        seq = pd.read_parquet(fname)
        #seq = seq[seq.dlq_adj <= 6 + 12]
        #seq = seq.sort_values(['loan_id', 'yyyymm']).reset_index(drop=True)
        seq_cudf = cudf.DataFrame.from_pandas(seq)
        seq_cudf = seq_cudf[seq_cudf.dlq_adj <= 6 + 12]
        seq_cudf = seq_cudf.sort_values(['loan_id', 'yyyymm']).reset_index(drop=True)
        seq = seq_cudf.to_pandas()
        
        lid = seq.loan_id.to_numpy()
        lid_idx = np.concatenate((np.array([0]), np.where(lid[:-1]!=lid[1:])[0]+1, np.array([len(lid)])))

        lid_to_seq_idx.append(pd.DataFrame({'loan_id':lid[lid_idx[:-1]], 'chunk_id': i, 'seq_idx_begin':lid_idx[:-1], 'seq_idx_end':lid_idx[1:]}))
        
        if includeLoanID:
            seq_numpy[i] = seq[['loan_id', 'default_1y', 'yyyymm', 'dlq_adj', 'age', 'int_rate', 'current_upb_norm']].to_numpy(dtype=np.float64)
        else:
            seq_numpy[i] = seq[[           'default_1y', 'yyyymm', 'dlq_adj', 'age', 'int_rate', 'current_upb_norm']].to_numpy(dtype=np.float32)
        del seq
        del seq_cudf
        gc.collect()
    
    print('concatenating lid_to_seq_idx')
    lid_to_seq_idx = pd.concat(lid_to_seq_idx).sort_values('loan_id')
    
    return lid_to_seq_idx, seq_numpy

def convertDataset(data_path):
    acquistion_fname = '/fnm_input_acq_parquet'
    sequence_fname = '/fnm_input_seq_parquet*'

    acquisition_nname = '/fnm_input_acq.npy'
    sequence_nname = '/fnm_input_seq_{}.npy'
    idx_to_seq_nname = '/fnm_input_idx_to_seq.npy'

    print('Data path: {}'.format(data_path))
    print('Acquistion parquet: {}'.format(data_path + acquistion_fname))
    print('Sequence parquet: {}'.format(data_path + sequence_fname))

    print("File found {}: {}".format(data_path + acquistion_fname, \
        os.path.exists(data_path + acquistion_fname)))

    seq_files = sorted([f for f in glob.glob(data_path + sequence_fname, recursive=False)])
    for f in seq_files:
        print('Sequence chunk found: {}'.format(f))

    print('Acquisition numpy: {}'.format(data_path + acquisition_nname))
    print('Sequence numpy: {}'.format(data_path + sequence_nname))
    print('Index to Sequence Index numpy: {}'.format(data_path + idx_to_seq_nname))
    
    acq_numpy = convertAcq(data_path + acquistion_fname)
    lid_to_seq_idx, seq_numpy = convertSeq(seq_files, includeLoanID = False)
    idx_to_seq = lid_to_seq_idx[['chunk_id', 'seq_idx_begin', 'seq_idx_end', 'loan_id']].to_numpy()
    
    np.save(data_path + acquisition_nname, acq_numpy, allow_pickle=False, fix_imports=False)
    np.save(data_path + idx_to_seq_nname, idx_to_seq, allow_pickle=False, fix_imports=False)
    for chunk_idx, seq_numpy_chunk in enumerate(seq_numpy):
        np.save(data_path + sequence_nname.format(chunk_idx), seq_numpy[chunk_idx], allow_pickle=False, fix_imports=False)

if __name__ == "__main__":
    convertDataset(data_path = '/home/user/notebooks/data/test/parquet')
    convertDataset(data_path = '/home/user/notebooks/data/train/parquet')
    convertDataset(data_path = '/home/user/notebooks/data/valid/parquet')