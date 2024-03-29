--    row_number() over (order by loan_id, rpt_period) as item_id,
drop table fnm_input_seq;
create table fnm_input_seq as
select
    a.loan_id,
    rpt_period,
    train_valid_test_ind,
    r,
    servicer_id,
    a.def_ind,
    load_period_cnt,
    default_period,
    default_dist,
    default_1y,
    dlq,
    age,
    int_rate,
    current_upb,
    months_to_maturity,
    msa,
    modification_flag
from fnm_collect_loans a join fnm_loan_split b on a.loan_id = b.loan_id
;
