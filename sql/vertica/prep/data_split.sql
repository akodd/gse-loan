
-- TODO: change to stratified sample

/*
Minimum requirement that can be relaxed in the future:
- 12 months training period 
- 13 months of prediction period

parameters:
- training size 70%
- validation size 10%
- testing size 20%

- 12 months LSTM trainings
- 13 months LSTM prediction
*/

-- drop table fnm_input_loans;
-- create table fnm_input_loans as
--     select
--         loan_id,
--         count(*) as observation_length
--     from fnm_prf
--     group by loan_id
--     having count(*) > 12 + 13
-- ;
-- 
-- drop table fnm_split;
-- create table fnm_split as 
--     with random_split as (
--         select
--             loan_id,
--             random() as rnd
--         from fnm_acq
--     ),
--     set_split as (
--         select
--             a.loan_id,
--             observation_length,
--             case
--                 when rnd <= 0.7 then 0 -- train
--                 when 0.7 < rnd and rnd <= 0.8 then 1 --valid
--                 else 2 --testing
--             end train_ind
--         from random_split a join fnm_input_loans b on a.loan_id = b.loan_id
--     )
--     select
--         loan_id,
--         observation_length,
--         row_number() over (partition by train_ind) as loan_inx,
--         train_ind
--     from set_split
-- ;
-- 
-- drop table loan_rpt_inx;
-- create table loan_rpt_inx as
--     with add_row_num as (
--         select
--             a.loan_id,
--             a.rpt_period,
--             b.loan_inx,
--             row_number() over (partition by a.loan_id order by a.rpt_period) as period_inx,
--             b.observation_length,
--             b.train_ind
--         from fnm_prf a join fnm_split b on a.loan_id = b.loan_id
--     )
--     select
--         *
--     from add_row_num
--     where observation_length - period_inx >= 12 + 13 /* 12 to train + 13 to predict */    
-- ;
 
drop table fnm_enum_train;
create table fnm_enum_train as
    with dataset as (
        select
            loan_id,
            rpt_period,
            row_number() over (order by loan_inx, period_inx asc) as seq_inx,
            loan_inx,
            period_inx,
            observation_length
        from loan_rpt_inx
        where train_ind = 0
    )
    select
        loan_id,
        rpt_period,
        seq_inx,
        loan_inx,
        period_inx,
        observation_length
    from dataset
    order by seq_inx
;

-- drop table fnm_enum_valid;
-- create table fnm_enum_valid as
--     select
--         loan_id,
--         rpt_period,
--         row_number() over (order by loan_inx, period_inx asc) as seq_inx,
--         loan_inx,
--         period_inx,
--         observation_length
--     from loan_rpt_inx
--     where train_ind = 1
-- ;
-- 
-- drop table fnm_enum_test;
-- create table fnm_enum_test as
--     select
--         loan_id,
--         rpt_period,
--         row_number() over (order by loan_inx, period_inx asc) as seq_inx,
--         loan_inx,
--         period_inx,
--         observation_length
--     from loan_rpt_inx
--     where train_ind = 2
-- ;

