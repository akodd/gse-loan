#!/usr/bin/python

import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Location and chunk number')
parser.add_argument('--dir', action='store', dest='datadir', \
    help='location for the data directory')
# parser.add_argument('--chunk', action='store', dest='chunk', \
#     help='chunk number 0 - 10')

res = parser.parse_args()
data_dir = res.datadir


parquet_file="{}/fnm_input_seq_{}".format(data_dir, chunk_number)

print('Reading file: ' +parquet_file)
fnm_input_seq_chunk = pd.read_parquet(parquet_file)

