drop table fnm_pd;
create table fnm_pd as
with 
    default_period_iden as ( --defaulted loans only
        select
            loan_id,
            min(rpt_period) as default_period
        from fnm_pre_pd_2
        where def_ind = 1 
        group by loan_id
    )
select
    a.loan_id,
    a.rpt_period,
    a.servicer_id,
    a.def_ind,
    a.load_period_cnt,
    b.default_period,
    nvl(months_between(rpt_period, default_period), -9999) as default_dist,
    a.dlq,
    a.age,
    a.int_rate,
    a.current_upb,
    a.months_to_maturity,
    a.msa,
    a.modification_flag
from fnm_pre_pd_2 a left join default_period_iden b
    on a.loan_id = b.loan_id
;