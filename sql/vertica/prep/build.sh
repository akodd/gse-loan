#!/bin/bash

sql_scripts=(
    fnm_input_loans 
    fnm_enum_train
    fnm_enum_test
    fnm_enum_valid
)

for sql_script in "${sql_scripts[@]}"
do
    echo "Running script: ${sql_script}.sql"
    /opt/vertica/bin/vsql -e -U dbadmin -f /home/dbadmin/gse-loan/sql/vertica/prep/${sql_script}.sql
done