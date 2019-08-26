#!/usr/bin/python

import pandas as pd
import numpy as np
import glob
import gc
import cudf
import time

def convertAcq(acq_fname, includeLoanID = False):
    acq = pd.read_parquet(acq_fname)
    acq = acq.sort_values('loan_id').reset_index(drop=True)
    acq = acq.fillna(0)
    if includeLoanID:
        acq_numpy = acq[[
            'loan_id', 'acq_def_ind', 'state_id', 'purpose_id', 'mi_type_id', \
            'occupancy_status_id', 'product_type_id', 'property_type_id', \
            'seller_id', 'zip3_id']].to_numpy()
    else:
        acq_numpy = acq[[          
                        'acq_def_ind', 'state_id', 'purpose_id', 'mi_type_id', \
            'occupancy_status_id', 'product_type_id', 'property_type_id', \
            'seller_id', 'zip3_id']].to_numpy(dtype=np.int32)
    return acq_numpy

def convertSeq(path_src, path_dest, includeLoanID = False):
    # mapping of index or loan id to chunk and sequence offsets
    lid_to_seq_idx =  []     
    seq_files = sorted([f for f in glob.glob(path_src, recursive=False)])
    
    for i, fname in enumerate(seq_files):
        start = time.clock()
        print('processing seq_numpy[{}]: {}'.format(i, fname))
        seq = pd.read_parquet(fname)
        seq_cudf = cudf.DataFrame.from_pandas(seq)
        seq_cudf = seq_cudf[seq_cudf.default_dist <= 11]
        seq_cudf = seq_cudf.sort_values(['loan_id', 'yyyymm']).reset_index(drop=True)
        seq = seq_cudf.to_pandas()
        
        lid = seq.loan_id.to_numpy()
        lid_idx = np.concatenate((
            np.array([0]), 
            np.where(lid[:-1]!=lid[1:])[0]+1, 
            np.array([len(lid)])
        ))

        lid_to_seq_idx.append(pd.DataFrame({
            'loan_id':lid[lid_idx[:-1]], 
            'chunk_id': i, 
            'seq_idx_begin':lid_idx[:-1], 
            'seq_idx_end':lid_idx[1:]
        }))
        
        if includeLoanID:
            seq_numpy = seq[[
                'loan_id', 'default_1y', 'yyyymm', 'dlq_adj', 'age', 'int_rate', \
                'current_upb_norm', 'msa_id', 'servicer_id'
            ]].to_numpy(dtype=np.float64)
        else:
            seq_numpy = seq[[           
                           'default_1y', 'yyyymm', 'dlq_adj', 'age', 'int_rate', \
                'current_upb_norm', 'msa_id', 'servicer_id'
            ]].to_numpy(dtype=np.float32)
        
        
        chunk_nname = path_dest.format(fname.split('/')[-1].split('_')[-1])
        print('saving: {}'.format(chunk_nname))
        np.save(chunk_nname, seq_numpy, allow_pickle=False, fix_imports=False)
        
        del seq
        del seq_cudf
        gc.collect()
    
    print('concatenating lid_to_seq_idx')
    lid_to_seq_idx = pd.concat(lid_to_seq_idx).sort_values('loan_id')
    
    return lid_to_seq_idx

def convertDataset(parquet_path, save_path):
    acquistion_fname = parquet_path + '/fnm_input_acq_parquet'
    sequence_fname = parquet_path + '/fnm_input_seq_parquet*'

    acquisition_nname = save_path + '/fnm_input_acq.npy'
    sequence_nname = save_path + '/fnm_input_seq_{}.npy'
    idx_to_seq_nname = save_path + '/fnm_input_idx_to_seq.npy'

    print('Data path: {}'.format(parquet_path))
    print('Acquistion parquet: {}'.format(acquistion_fname))
    print('Sequence parquet: {}'.format(sequence_fname))

    seq_files = sorted([f for f in glob.glob(sequence_fname, recursive=False)])
    for f in seq_files:
        print('Sequence chunk found: {}'.format(f))

    print('Acquisition numpy: {}'.format(acquisition_nname))
    print('Sequence numpy: {}'.format(sequence_nname))
    print('Index to Sequence Index numpy: {}'.format(idx_to_seq_nname))
    
    acq_numpy = convertAcq(acquistion_fname)
    lid_to_seq_idx = convertSeq(sequence_fname, sequence_nname, includeLoanID = False)
    idx_to_seq = lid_to_seq_idx[[
        'chunk_id', 'seq_idx_begin', 'seq_idx_end', 'loan_id'
        ]].to_numpy(dtype=np.int64)
    
    print('Saving: {}'.format(acquisition_nname))
    np.save(acquisition_nname, acq_numpy, allow_pickle=False, fix_imports=False)
    print('Saving: {}'.format(idx_to_seq_nname))
    np.save(idx_to_seq_nname, idx_to_seq, allow_pickle=False, fix_imports=False)
    

if __name__ == "__main__":
    convertDataset(
        parquet_path = '/home/user/notebooks/data/valid/parquet', 
        save_path='/home/user/notebooks/data/valid'
    )

    convertDataset(
        parquet_path = '/home/user/notebooks/data/test/parquet', 
        save_path='/home/user/notebooks/data/test'
    )
    convertDataset(
        parquet_path = '/home/user/notebooks/data/train/parquet', 
        save_path='/home/user/notebooks/data/train'
    )
    