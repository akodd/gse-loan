

-- TODO: change to stratified sample

/*
Minimum requirement that can be relaxed in the future:
- 12 months training period 
- 13 months of prediction period
*/


drop table fnm_enum_testing;
create table fnm_enum_testing as
    with loan_rpt_inx as (
        select
            a.loan_id,
            a.rpt_period,
            b.loan_inx,
            row_number() over (partition by a.loan_id order by a.rpt_period) as period_inx,
            b.observation_length
        from fnm_prf a join fnm_split b on a.loan_id = b.loan_id
        where b.training_ind = 0
    )
    select
        loan_id,
        rpt_period,
        loan_inx,
        period_inx,
        observation_length,
        row_number() over (partition by loan_inx order by period_inx) as train_inx
    from loan_rpt_inx
    where observation_length - period_inx >= 12 + 13 /* 12 to train + 13 to predict */
;
