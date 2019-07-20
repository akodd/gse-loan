
drop table fnm_pre_pd_2;
create table fnm_pre_pd_2 as
select
    loan_id,
    rpt_period,
    servicer_id,
    max(def_ind) over (
        partition by loan_id order by rpt_period 
        range between unbounded preceding and current row
    ) as def_ind,
    row_number() over (partition by loan_id order by rpt_period) as load_period_cnt,
    dlq,
    age,
    int_rate,
    current_upb,
    months_to_maturity,
    msa,
    modification_flag
from fnm_pre_pd_1
;
