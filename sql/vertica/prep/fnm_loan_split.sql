drop table fnm_loan_split;
create table fnm_loan_split as
with
collect_loans as (
    select
        loan_id,
        max(def_ind) as def_ind,
        random() r
    from fnm_collect_loans
    group by loan_id
),
split_loans as (
    select
        loan_id,
        def_ind,
        r,
        case
            when r <=0.8 then 0 -- training
            when 0.8<r and r<=0.9 then 1-- validation
            else 2
        end train_valid_test_ind
    from collect_loans
)
select
    loan_id,
    r,
    train_valid_test_ind,
    row_number() over (partition by train_valid_test_ind order by r) loan_item_id
from split_loans
;