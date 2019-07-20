drop table fnm_collect_loans;
/*
    We assume minimum 12 periods training sequence
    We sssume minimum 12 periods prediction sequence
*/
create table fnm_collect_loans as
select
    loan_id,
    rpt_period,
    servicer_id,
    def_ind,
    load_period_cnt,
    default_period,
    default_dist,
    case
        when default_dist >= -11 then 1
        else 0
    end as default_1y,
    dlq,
    age,
    int_rate,
    current_upb,
    months_to_maturity,
    msa,
    modification_flag
from fnm_pd
where loan_id in (
    -- collect sequences with 24 elements or longerr
    select  
        loan_id
    from fnm_pd
    -- remove accouts long in default
    where default_dist <= 11
    group by loan_id
    -- training + prediction and removes left censored
    having max(load_period_cnt) >= 12 + 12 
)
;


