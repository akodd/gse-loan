drop table fnm_export_seq;
create table fnm_export_seq as
with pack_sequence as (
    select
        train_valid_test_ind,
        r,
        cast(loan_id as int) loan_id,
        cast(year(rpt_period)*100+month(rpt_period) as int) as yyyymm,
        cast(nvl(servicer_id, 0) as int) servicer_id,
        cast(default_1y as int) default_1y,
        cast(dlq as int) dlq,
        cast(default_dist as int) default_dist,
        cast(age as int) age,
        cast(int_rate as float) int_rate,
        cast(current_upb as float) current_upb,
        -- cast(months_to_maturity as int) months_to_maturity, # leave for now
        cast(nvl(msa_id, 0) as int) msa_id,
        case 
            when modification_flag='Y' then 1
            else 0
        end modification_flag
    from fnm_input_seq left join fnm_msa using (msa)
    where loan_id not in (select loan_id from fnm_gapped_loans)
),
fill_forward as (
    select
        train_valid_test_ind,
        r,
        loan_id,
        yyyymm,
        servicer_id,
        default_1y,
        dlq,
        default_dist,
        last_value (age ignore nulls) over (
            partition by loan_id order by yyyymm
        ) as age,
        last_value (int_rate ignore nulls) over (
            partition by loan_id order by yyyymm
        ) as int_rate,
        last_value (current_upb ignore nulls) over (
            partition by loan_id order by yyyymm
        ) as current_upb,
        last_value (msa_id ignore nulls) over (
            partition by loan_id order by yyyymm
        ) as msa_id,
        modification_flag
    from pack_sequence
),
fill_backward as (
    select
        train_valid_test_ind,
        r,
        loan_id,
        yyyymm,
        servicer_id,
        default_1y,
        dlq,
        default_dist,
        first_value (age ignore nulls) over (
            partition by loan_id order by yyyymm
            ROWS BETWEEN current row AND UNBOUNDED FOLLOWING
        ) as age,
        first_value (int_rate ignore nulls) over (
            partition by loan_id order by yyyymm
            ROWS BETWEEN current row AND UNBOUNDED FOLLOWING
        ) as int_rate,
        first_value (current_upb ignore nulls) over (
            partition by loan_id order by yyyymm
            ROWS BETWEEN current row AND UNBOUNDED FOLLOWING
        ) as current_upb,
        first_value (msa_id ignore nulls) over (
            partition by loan_id order by yyyymm
            ROWS BETWEEN current row AND UNBOUNDED FOLLOWING
        ) as msa_id,
        modification_flag
    from fill_forward
),
mean_upb as (
    select
        avg(current_upb) as mean_upb,
        stddev(current_upb) as sd_upb
    from fnm_input_seq
)
select
    train_valid_test_ind,
    r,
    loan_id,
    yyyymm,
    servicer_id,
    default_1y,
    dlq,
    default_dist,
    case
        when default_dist > 0 then default_dist + 7
        else dlq
    end dlq_adj,
    age,
    int_rate,
    current_upb,
    (current_upb - mean_upb)/sd_upb as current_upb_norm,
    msa_id,
    modification_flag
from fill_backward, mean_upb
order by train_valid_test_ind, r, loan_id, yyyymm