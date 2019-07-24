#!/bin/bash


sql_stmt=$(<export_fnm_input_acq_train.sql)
docker-compose exec vertica /opt/vertica/bin/vsql -e -U dbadmin -c "${sql_stmt}"

echo "Moving Parquet file"
sudo mv \
    ../../../data-db/vertica/fnm_input_acq_train \
    ../../../data/fnm_input_acq_train

echo "Converting Paquet file into Feather file"
docker-compose exec jupyter-labn \
    /home/user/notebooks/sql/vertica/export/parquet_to_feather_acq.py

echo "Cleanup Parquet file ../../../data/fnm_input_acq_train"
sudo rm -rf ../../../data/fnm_input_acq_train