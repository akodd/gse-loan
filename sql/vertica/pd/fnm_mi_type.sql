drop table fnm_mi_type;
create table fnm_mi_type as
    with mi_types as (
        select
            distinct mi_type
        from fnm_acq
        where mi_type is not null
    )
    select
        row_number() over (order by mi_type) as mi_type_id,
        mi_type
    from mi_types
;
