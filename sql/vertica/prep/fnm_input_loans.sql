drop table fnm_input_loans;
create table fnm_input_loans as
    select
        loan_id,
        count(*) as observation_length
    from fnm_prf
    group by loan_id
    having count(*) > 12 + 13
;