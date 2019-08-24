#!/bin/bash

VERTICA_EXPORT_DIR='/home/dbadmin/docker'

TRAIN_TARGET_DIR='../../../data/train/parquet'
VALID_TARGET_DIR='../../../data/valid/parquet'
TEST_TARGET_DIR='../../../data/test/parquet'

mkdir -p ${TRAIN_TARGET_DIR}
mkdir -p ${VALID_TARGET_DIR}
mkdir -p ${TEST_TARGET_DIR}

make_seq_sql () {
    local sql_stmt=$(<export_fnm_input_seq.sql)
    local parquet_file=$1
    local tvt_arg=$2
    local lower_boundary=$3
    local upper_boundary=$4

    local sql_stmt=${sql_stmt/CHUNK_DIR/${parquet_file}}
    local sql_stmt=${sql_stmt/LOWER_BOUNDARY/${lower_boundary}}
    local sql_stmt=${sql_stmt/UPPER_BOUNDARY/${upper_boundary}} 
    local sql_stmt=${sql_stmt/TVT_SLICE/${tvt_arg}} 

    echo "${sql_stmt}"
}

run_sql () {
    docker-compose exec vertica /opt/vertica/bin/vsql -U dbadmin -c "$1"
}

# Extract Training Data
echo 'Exporting training data'
seq_cuts=(0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 \
          0.22 0.24 0.26 0.28 0.30 0.32 0.34 0.36 0.38 0.40 0.42 \
          0.44 0.46 0.48 0.50 0.52 0.54 0.56 0.58 0.60 0.62 0.64 \
          0.66 0.68 0.70 0.72 0.74 0.76 0.78 0.80)
for (( i=1; i<${#seq_cuts[@]}; i++ ))
do
    chunk=$(($i-1))
    VERTICA_CHUNK_NAME="${VERTICA_EXPORT_DIR}/fnm_input_seq_parquet_${chunk}"
    sql_stmt=$( make_seq_sql $VERTICA_CHUNK_NAME 0 ${seq_cuts[i-1]} ${seq_cuts[i]} )
    echo `date +'%Y-%m-%d %H:%M:%S'`, "Exporting: ${VERTICA_CHUNK_NAME}"
    run_sql "${sql_stmt}"
    echo "Moving Parquet file to ${TRAIN_TARGET_DIR}"
    sudo mv \
        ../../../data-db/vertica/fnm_input_seq_parquet_${chunk} \
        ${TRAIN_TARGET_DIR}/fnm_input_seq_parquet_${seq_cuts[i-1]}
done

# Extract Validation Data
echo 'Exporting validation data'
seq_cuts=(0.80 0.82 0.84 0.86 0.88 0.90)
for (( i=1; i<${#seq_cuts[@]}; i++ ))
do
    chunk=$(($i-1))
    VERTICA_CHUNK_NAME="${VERTICA_EXPORT_DIR}/fnm_input_seq_parquet_${chunk}"
    sql_stmt=$( make_seq_sql $VERTICA_CHUNK_NAME 1 ${seq_cuts[i-1]} ${seq_cuts[i]} ) #1 is for validation
    echo `date +'%Y-%m-%d %H:%M:%S'`, "Exporting: ${VERTICA_CHUNK_NAME}"
    run_sql "${sql_stmt}"
    echo "Moving Parquet file to ${VALID_TARGET_DIR}"
    sudo mv \
        ../../../data-db/vertica/fnm_input_seq_parquet_${chunk} \
        ${VALID_TARGET_DIR}/fnm_input_seq_parquet_${seq_cuts[i-1]}
done

# Extract Testing Data
echo 'Exporting testing data'
seq_cuts=(0.90 0.92 0.94 0.96 0.98 1.00)
for (( i=1; i<${#seq_cuts[@]}; i++ ))
do
    chunk=$(($i-1))
    VERTICA_CHUNK_NAME="${VERTICA_EXPORT_DIR}/fnm_input_seq_parquet_${chunk}"
    sql_stmt=$( make_seq_sql $VERTICA_CHUNK_NAME 2 ${seq_cuts[i-1]} ${seq_cuts[i]} ) #2 is for validation
    echo `date +'%Y-%m-%d %H:%M:%S'`, "Exporting: ${VERTICA_CHUNK_NAME}"
    run_sql "${sql_stmt}"
    echo "Moving Parquet file to ${TEST_TARGET_DIR}"
    sudo mv \
        ../../../data-db/vertica/fnm_input_seq_parquet_${chunk} \
        ${TEST_TARGET_DIR}/fnm_input_seq_parquet_${seq_cuts[i-1]}
done