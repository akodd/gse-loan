-- TODO: change to stratified sample

/*
Minimum requirement that can be relaxed in the future:
- 12 months training period 
- 13 months of prediction period
*/


drop table fnm_input_loans;
create table fnm_input_loans as
    select
        loan_id,
        count(*) as observation_length
    from fnm_prf
    group by loan_id
    having count(*) > 25
;


drop table fnm_split;
create table fnm_split as 
    with random_split as (
        select
            loan_id,
            random() as rnd
        from fnm_acq
    )
    select
        row_number() over (order by a.loan_id) as loan_inx,
        a.loan_id,
        observation_length,
        case
            when rnd <= 0.7 then 0 -- train
            when 0.7 < rnd and rnd <= 0.8 then 1 --valid
            else 2 --testing
        end training_ind
    from random_split a join fnm_input_loans b on a.loan_id = b.loan_id
;


drop table fnm_enum_testing;
create table fnm_enum_testing as
    select
        a.loan_id,
        a.rpt_period,
        b.loan_inx,
        row_number() over (partition by a.loan_id order by a.rpt_period) as period_inx,
        b.observation_length
    from fnm_prf a join fnm_split b on a.loan_id = b.loan_id
    where b.training_ind = 0
;
