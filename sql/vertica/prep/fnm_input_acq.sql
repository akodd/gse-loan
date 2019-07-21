drop table fnm_input_acq;
-- We keep this table as is and cache it in memory
create table fnm_input_acq as
select
    c.loan_id,
    def_ind,
    r,
    s.loan_item_id,
    train_valid_test_ind,
    a1.state_id,
    a2.purpose_id,
    a3.mi_type_id,
    a4.occupancy_status_id,
    a5.product_type_id,
    a6.property_type_id,
    a7.seller_id,
    a8.zip3_id
from fnm_acq c  left join fnm_state a1 on c.state = a1.state
                left join fnm_loan_purpose a2 on c.purpose = a2.purpose
                left join fnm_mi_type a3 on c.mi_type = a3.mi_type
                left join fnm_occupancy_status a4 on c.occupancy_status = a4.occupancy_status
                left join fnm_product_type a5 on c.product_type = a5.product_type
                left join fnm_property_type a6 on c.property_type = a6.property_type
                left join fnm_seller_name a7 on c.seller_name = a7.seller_name
                left join fnm_zip3 a8 on c.zip3 = a8.zip3
                join fnm_loan_split s on s.loan_id = c.loan_id
;