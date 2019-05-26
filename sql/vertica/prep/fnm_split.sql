drop table fnm_split;
create table fnm_split as 
    with random_split as (
        select
            loan_id,
            random() as rnd
        from fnm_acq
    ),
    set_split as (
        select
            a.loan_id,
            observation_length,
            case
                when rnd <= 0.7 then 0 -- train
                when 0.7 < rnd and rnd <= 0.8 then 1 --valid
                else 2 --testing
            end train_ind
        from random_split a join fnm_input_loans b on a.loan_id = b.loan_id
    )
    select
        loan_id,
        observation_length,
        row_number() over (partition by train_ind) as loan_inx,
        train_ind
    from set_split
;