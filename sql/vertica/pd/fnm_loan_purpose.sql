drop table fnm_loan_purpose;
create table fnm_loan_purpose as
    with loan_purposes as (
        select
            distinct purpose
        from fnm_acq
        where purpose is not null
    )
    select
        row_number() over (order by purpose) as purpose_id,
        purpose
    from loan_purposes
;