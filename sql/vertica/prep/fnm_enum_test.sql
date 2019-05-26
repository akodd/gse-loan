drop table fnm_enum_test;
create table fnm_enum_test as
    select
        loan_id,
        row_number() over (order by loan_id) as loan_inx,
        observation_length,
        state_id,
        purpose_id,
        mi_type_id,
        occupancy_status_id,
        product_type_id,
        property_type_id,
        seller_id,
        zip3_id
    from fnm_input_loans
    where train_ind = 2
    order by loan_inx
;