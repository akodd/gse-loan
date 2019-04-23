create table fnm_def1 as
select
    loan_id,
    rpt_period,
    last_value(servicer_name ignore nulls) over (partition by loan_id order by rpt_period) as servicer_name_fill,
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
    age,
    int_rate,
    current_upb,
    months_to_maturity,
    msa
from fnm_prf
;

--create table fnm_def as
--with
--    loan_dlq as (
--        select 
--            loan_id,
--            rpt_period,
--            case
--                when trim(del) = 'X' then 0
--                else cast(del as int)
--            end as dlq
--        from fnm_prf
--    ),
--    def_acc as (
--        select 
--            loan_id,
--            min(rpt_period) as rpt_period_def
--        from loan_dlq
--        where dlq >= 7
--        group by loan_id
--    )
--    select
--        a.loan_id,
--        rpt_period,
--        rpt_period_def,
--        dlq,
--        age,
--        int_rate,
--        current_upb,
--        months_to_maturity,
--        case
--            when rpt_period >= rpt_period_def then 1
--            else 0
--        end as def_ind
--    from loan_dlq a left join def_acc b on a.loan_id = b.loan_id
--                    join fnm_prf c on a.loan_id = c.load_id
--;