#!/bin/bash

VERTICA_EXPORT_DIR='/home/dbadmin/docker'

TRAIN_TARGET_DIR='../../../data/train/parquet'
VALID_TARGET_DIR='../../../data/valid/parquet'
TEST_TARGET_DIR='../../../data/test/parquet'

mkdir -p ${TRAIN_TARGET_DIR}
mkdir -p ${VALID_TARGET_DIR}
mkdir -p ${TEST_TARGET_DIR}

VERTICA_PARQUET="${VERTICA_EXPORT_DIR}/fnm_input_acq_parquet"

make_seq_sql () {
    local sql_stmt=$(<export_fnm_input_acq.sql)
    local tvt_arg=$1
    local sql_stmt=${sql_stmt/TVT_SLICE/${tvt_arg}} 
    echo "${sql_stmt}"
}

run_sql () {
    docker-compose exec vertica /opt/vertica/bin/vsql -U dbadmin -c "$1"
}

# Extracting Training Data
sql_stmt=$( make_seq_sql 0 )
echo `date +'%Y-%m-%d %H:%M:%S'`, "Exporting: ${VERTICA_PARQUET}"
run_sql "${sql_stmt}"

echo "Moving Parquet file to ${TRAIN_TARGET_DIR}"
sudo mv \
    ../../../data-db/vertica/fnm_input_acq_parquet \
    ${TRAIN_TARGET_DIR}/fnm_input_acq_parquet

# Extracting Validation Data
sql_stmt=$( make_seq_sql 1 )
echo `date +'%Y-%m-%d %H:%M:%S'`, "Exporting: ${VERTICA_PARQUET}"
run_sql "${sql_stmt}"

echo "Moving Parquet file to ${VALID_TARGET_DIR}"
sudo mv \
    ../../../data-db/vertica/fnm_input_acq_parquet \
    ${VALID_TARGET_DIR}/fnm_input_acq_parquet


# Extracting Testing Data
sql_stmt=$( make_seq_sql 2 )
echo `date +'%Y-%m-%d %H:%M:%S'`, "Exporting: ${VERTICA_PARQUET}"
run_sql "${sql_stmt}"

echo "Moving Parquet file to ${TEST_TARGET_DIR}"
sudo mv \
    ../../../data-db/vertica/fnm_input_acq_parquet \
    ${TEST_TARGET_DIR}/fnm_input_acq_parquet

sudo chmod a+r ${TEST_TARGET_DIR}/fnm_input_acq_parquet