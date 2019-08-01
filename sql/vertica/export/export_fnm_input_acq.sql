export to parquet (directory='/home/dbadmin/docker/fnm_input_acq_parquet')
as
select
    cast (loan_id                      as int)  as loan_id             , 
    cast (r                            as float) as r                   , 
    cast (nvl(loan_item_id, 0)         as int)  as loan_item_id        , 
    cast (nvl(state_id, 0)             as int)  as state_id            , 
    cast (nvl(purpose_id, 0)           as int)  as purpose_id          , 
    cast (nvl(mi_type_id, 0)           as int)  as mi_type_id          , 
    cast (nvl(occupancy_status_id, 0)  as int)  as occupancy_status_id , 
    cast (nvl(product_type_id, 0)      as int)  as product_type_id     , 
    cast (nvl(property_type_id, 0)     as int)  as property_type_id    , 
    cast (nvl(seller_id, 0)            as int)  as seller_id           , 
    cast (nvl(zip3_id, 0)              as int)  as zip3_id             
from fnm_input_acq
where train_valid_test_ind = TVT_SLICE
;
