#!/usr/bin/python

import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Expect chunk number')
parser.add_argument('--chunk', action='store', dest='chunk', help='chunk number 0 - 10')

res = parser.parse_args()
chunk_number = res.chunk
parquet_file="/home/user/notebooks/data/fnm_input_seq_train_{}".format(chunk_number)
feather_file="/home/user/notebooks/data/fnm_input_seq_train_{}.feather".format(chunk_number)

print('Reading file: ' +parquet_file)

fnm_input_seq_chunk = pd.read_parquet(parquet_file)
fnm_input_seq_chunk = fnm_input_seq_chunk.astype({
    'yyyymm': 'int32',
    'servicer_id': 'int8',
    'default_1y': 'uint8',
    'dlq': 'uint8',
    'age': 'uint8',
    'int_rate': 'float32',
    'msa': 'int16',
    'modification_flag': 'uint8'
})

print('Writing file: ' +feather_file)
fnm_input_seq_chunk.to_feather(feather_file)
