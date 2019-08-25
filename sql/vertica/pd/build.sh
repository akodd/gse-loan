#!/bin/bash

sql_scripts=(
    fnm_seller_name 
    fnm_servicer_name 
    fnm_loan_purpose
    fnm_property_type
    fnm_occupancy_status
    fnm_state
    fnm_zip3
    fnm_product_type
    fnm_mi_type
    fnm_msa
    fnm_pd
)

for sql_script in "${sql_scripts[@]}"
do
    echo "Running script: ${sql_script}.sql"
    docker-compose exec vertica /opt/vertica/bin/vsql -e -U dbadmin \
        -f /home/dbadmin/gse-loan/sql/vertica/pd/${sql_script}.sql
done