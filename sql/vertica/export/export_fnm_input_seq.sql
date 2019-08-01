export to parquet (directory='CHUNK_DIR')
as
with pack_sequence as (
    select
        cast(loan_id as int) loan_id,
        cast(year(rpt_period)*100+month(rpt_period) as int) as yyyymm,
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
    where train_valid_test_ind = TVT_SLICE
        and LOWER_BOUNDARY <= r 
        and r < UPPER_BOUNDARY 
),
fill_forward as (
    select
        loan_id,
        yyyymm,
        servicer_id,
        default_1y,
        dlq,
        last_value (age ignore nulls) over (
            partition by loan_id order by yyyymm
        ) as age,
        last_value (int_rate ignore nulls) over (
            partition by loan_id order by yyyymm
        ) as int_rate,
        last_value (current_upb ignore nulls) over (
            partition by loan_id order by yyyymm
        ) as current_upb,
        last_value (msa ignore nulls) over (
            partition by loan_id order by yyyymm
        ) as msa,
        modification_flag
    from pack_sequence
)
select
    loan_id,
    yyyymm,
    servicer_id,
    default_1y,
    dlq,
    first_value (age ignore nulls) over (
        partition by loan_id order by yyyymm
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) as age,
    first_value (int_rate ignore nulls) over (
        partition by loan_id order by yyyymm
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) as int_rate,
    first_value (current_upb ignore nulls) over (
        partition by loan_id order by yyyymm
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) as current_upb,
    first_value (msa ignore nulls) over (
        partition by loan_id order by yyyymm
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) as msa,
    modification_flag
from fill_forward
;
