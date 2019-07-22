export to parquet (directory='/home/dbadmin/docker/fnm_input_seq_train_CHUNK')
as
select
    cast(loan_id as int) loan_id,
    rpt_period,
    cast(nvl(servicer_id, 0) as int) servicer_id,
    cast(default_1y as int) default_1y,
    cast(dlq as int) dlq,
    cast(age as int) age,
    cast(int_rate as float) int_rate,
    cast(current_upb as float) current_upb,
    -- cast(months_to_maturity as int) months_to_maturity, # leave for now
    cast(msa as int) msa,
    case 
        when modification_flag='Y' then 1
        else 0
    end modification_flag
from fnm_input_seq
where train_valid_test_ind = 0 
    and LOWER_BOUNDARY <= r 
    and r < UPPER_BOUNDARY -- 10% of training data
order by loan_id, rpt_period
;
