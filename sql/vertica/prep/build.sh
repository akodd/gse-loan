#!/bin/bash

sql_scripts=(
    # fnm_input_loans 
    # fnm_split
    # fnm_enum_testing
    data_split
)

for sql_script in "${sql_scripts[@]}"
do
    echo "Running script: ${sql_script}.sql"
    /opt/vertica/bin/vsql -U dbadmin -f /home/dbadmin/gse-loan/sql/vertica/prep/${sql_script}.sql
done