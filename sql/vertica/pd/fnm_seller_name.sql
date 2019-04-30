drop table fnm_seller_name;
create table fnm_seller_name as
    with seller_names as (
        select
            distinct seller_name
        from fnm_acq
    )
    select
        row_number() over (order by seller_name) as seller_id,
        seller_name
    from seller_names
;