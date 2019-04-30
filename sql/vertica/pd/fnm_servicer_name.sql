drop table fnm_servicer_name;
create table fnm_servicer_name as
    with servicer_names as (
        select
            distinct servicer_name
        from fnm_prf
        where servicer_name is not null
    )
    select
        row_number() over (order by servicer_name) as servicer_id,
        servicer_name
    from servicer_names
;