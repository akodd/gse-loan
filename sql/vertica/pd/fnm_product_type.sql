drop table fnm_product_type;
create table fnm_product_type as
    with product_types as (
        select
            distinct product_type
        from fnm_acq
        where product_type is not null
    )
    select
        row_number() over (order by product_type) as product_type_id,
        product_type
    from product_types
;