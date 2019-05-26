-- 1+ months LSTM trainings
-- 13 months LSTM prediction
drop table fnm_input_loans;
create table fnm_input_loans as
    with good_loans as (
        select
            loan_id,
            count(*) as observation_length
        from fnm_prf
        group by loan_id
        having count(*) > 1 + 13
    ),
    assign_rdn as (
        select
            loan_id,
            random() as rnd,
            observation_length
        from good_loans
    )
    select
        loan_id,
        observation_length,
        case
            when rnd <= 0.7 then 0 -- training
            when 0.7 < rnd and rnd <= 0.8 then 1 --validation
            else 2 --testing
        end train_ind
    from assign_rdn
;
