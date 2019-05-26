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
        a.loan_id,
        a.observation_length,
        case
            when rnd <= 0.7 then 0 -- training
            when 0.7 < rnd and rnd <= 0.8 then 1 --validation
            else 2 --testing
        end train_ind,
        a1.state_id,
        a2.purpose_id,
        a3.mi_type_id,
        a4.occupancy_status_id,
        a5.product_type_id,
        a6.property_type_id,
        a7.seller_id,
        a8.zip3_id
    from assign_rdn a join fnm_acq c on a.loan_id = c.loan_id
                      join fnm_state a1 on c.state = a1.state
                      join fnm_loan_purpose a2 on c.purpose = a2.purpose
                      join fnm_mi_type a3 on c.mi_type = a3.mi_type
                      join fnm_occupancy_status a4 on c.occupancy_status = a4.occupancy_status
                      join fnm_product_type a5 on c.product_type = a5.product_type
                      join fnm_property_type a6 on c.property_type = a6.property_type
                      join fnm_seller_name a7 on c.seller_name = a7.seller_name
                      join fnm_zip3 a8 on c.zip3 = a8.zip3
;
