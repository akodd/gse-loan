drop table fnm_zip3;
create table fnm_zip3 as
    with zip3s as (
        select
            distinct zip3
        from fnm_acq
        where zip3 is not null
    )
    select
        row_number() over (order by zip3) as zip3_id,
        zip3
    from zip3s
;
