#!/usr/bin/python

import pandas as pd

parquet_file="/home/user/notebooks/data/fnm_input_acq_train"
feather_file="/home/user/notebooks/data/fnm_input_acq_train.feather"

print('Reading file: ' +parquet_file)

fnm_input_acq_train = pd.read_parquet(parquet_file)
fnm_input_acq_train = fnm_input_acq_train.astype({
    'state_id': 'int8',
    'purpose_id': 'int8',
    'mi_type_id': 'int8',
    'occupancy_status_id': 'int8',
    'product_type_id': 'int8',
    'property_type_id': 'int8',
    'seller_id': 'int8',
    'zip3_id': 'int16'
})

print('Writing file: ' +feather_file)
fnm_input_acq_train.to_feather(feather_file)
