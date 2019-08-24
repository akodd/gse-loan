#!/bin/bash

sql_scripts=(
    fnm_collect_loans
    fnm_loan_split
    fnm_input_acq
    fnm_input_seq
    #fnm_add_pkeys.sql
    fnm_gapped_loans
)

for sql_script in "${sql_scripts[@]}"
do
    echo "Running script: ${sql_script}.sql"
    docker-compose exec vertica /opt/vertica/bin/vsql -e -U dbadmin -f /home/dbadmin/gse-loan/sql/vertica/prep/${sql_script}.sql
done