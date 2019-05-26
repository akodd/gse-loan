drop table fnm_enum_test;
create table fnm_enum_test as
    with dataset as (
        select
            loan_id,
            row_number() over () as loan_inx,
            observation_length
        from fnm_input_loans
        where train_ind = 2
    )
    select
        loan_id,
        loan_inx,
        observation_length
    from dataset
    order by loan_inx
;