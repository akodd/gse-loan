#!/bin/bash

VERTICA_EXPORT_DIR='/home/dbadmin/docker'

TRAIN_TARGET_DIR='../../../data/train'
VALID_TARGET_DIR='../../../data/valid'
TEST_TARGET_DIR='../../../data/test'

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
seq_cuts=(0.0 0.08 0.16 0.24 0.32 0.40 0.48 0.56 0.64 0.72 0.80)
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
        ${TRAIN_TARGET_DIR}/fnm_input_seq_parquet_${chunk}
done

# Extract Validation Data
echo 'Exporting validation data'
VERTICA_CHUNK_NAME="${VERTICA_EXPORT_DIR}/fnm_input_seq_parquet"
sql_stmt=$( make_seq_sql $VERTICA_CHUNK_NAME 1 0 1.1 ) #1 is for validation
echo `date +'%Y-%m-%d %H:%M:%S'`, "Exporting: ${VERTICA_CHUNK_NAME}"
run_sql "${sql_stmt}"
echo "Moving Parquet file to ${VALID_TARGET_DIR}"
sudo mv \
    ../../../data-db/vertica/fnm_input_seq_parquet \
    ${VALID_TARGET_DIR}/fnm_input_seq_parquet_0

# Extract Testing Data
echo 'Exporting testing data'
VERTICA_CHUNK_NAME="${VERTICA_EXPORT_DIR}/fnm_input_seq_parquet"
sql_stmt=$( make_seq_sql $VERTICA_CHUNK_NAME 2 0 1.1 ) #2 is for validation
echo `date +'%Y-%m-%d %H:%M:%S'`, "Exporting: ${VERTICA_CHUNK_NAME}"
run_sql "${sql_stmt}"
echo "Moving Parquet file to ${TEST_TARGET_DIR}"
sudo mv \
    ../../../data-db/vertica/fnm_input_seq_parquet \
    ${TEST_TARGET_DIR}/fnm_input_seq_parquet_0