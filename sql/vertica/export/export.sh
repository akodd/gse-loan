#!/bin/bash

# cuts small enough to keap feather files small (TODO: bug in feather size)
seq_cuts=(0.0 0.08) # 0.16 0.24 0.32 0.40 0.48 0.56 0.64 0.72 0.80)

for (( i=1; i<${#seq_cuts[@]}; i++ ))
do
    sql_stmt=$(<export_fnm_input_seq_train.sql)
    chunk=$(($i-1))
    sql_stmt=${sql_stmt/CHUNK/${chunk}}
    sql_stmt=${sql_stmt/LOWER_BOUNDARY/${seq_cuts[i-1]}}
    sql_stmt=${sql_stmt/UPPER_BOUNDARY/${seq_cuts[i]}}
    echo `date +'%Y-%m-%d %H:%M:%S'`, "Exporting chunk: ${chunk}"
    docker-compose exec vertica /opt/vertica/bin/vsql -e -U dbadmin -c "${sql_stmt}"
    echo "Moving Parquet file"
    sudo mv \
        ../../../data-db/vertica/fnm_input_seq_train_${chunk} \
        ../../../data/fnm_input_seq_train_${chunk}
    echo "Converting Paquet file into Feather file"
    docker-compose exec jupyter-labn \
        /home/user/notebooks/sql/vertica/export/parquet_to_feather.py --chunk ${chunk}
    echo "Cleanup Parquet file ../../../data/fnm_input_seq_train_${chunk}"
    sudo rm -rf ../../../data/fnm_input_seq_train_${chunk}
done