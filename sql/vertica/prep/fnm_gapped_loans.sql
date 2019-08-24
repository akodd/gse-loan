drop table fnm_gapped_loans;
create table fnm_gapped_loans as
with lagged as (
    select
        loan_id,
        rpt_period,
        lag(rpt_period) over (partition by loan_id order by rpt_period) rpt_period_lag1
    from fnm_input_seq
)
select
    distinct loan_id
from lagged
where months_between(rpt_period, rpt_period_lag1) != 1
;