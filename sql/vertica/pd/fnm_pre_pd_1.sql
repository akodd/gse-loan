
drop table fnm_pre_pd_1;
create table fnm_pre_pd_1 as
select
    p.loan_id,
    p.rpt_period,
    last_value(servicer_id ignore nulls) over (partition by p.loan_id order by rpt_period) as servicer_id,
    case
        when (case
                when trim(del) = 'X' then 0
                else cast(del as int)
            end) >= 7 then 1
        else 0
    end as def_ind,
    case
        when trim(del) = 'X' then 0
        else cast(del as int)
    end as dlq,
    p.age,
    p.int_rate,
    p.current_upb,
    p.months_to_maturity,
    p.msa,
    p.modification_flag
from fnm_prf p join fnm_acq a on p.loan_id = a.loan_id
               left join fnm_servicer_name sv on p.servicer_name = sv.servicer_name
;
