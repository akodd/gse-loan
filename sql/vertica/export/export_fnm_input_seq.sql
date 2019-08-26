export to parquet (directory='CHUNK_DIR')
as
select
    train_valid_test_ind,
    r,
    loan_id,
    yyyymm,
    servicer_id,
    default_1y,
    dlq,
    default_dist,
    dlq_adj,
    age,
    int_rate,
    current_upb,
    current_upb_norm,
    msa_id,
    modification_flag
from fnm_export_seq
where train_valid_test_ind = TVT_SLICE
        and LOWER_BOUNDARY <= r 
        and r < UPPER_BOUNDARY
;