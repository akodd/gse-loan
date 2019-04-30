drop table fnm_occupancy_status;
create table fnm_occupancy_status as
    with occupancy_statuss as (
        select
            distinct occupancy_status
        from fnm_acq
        where occupancy_status is not null
    )
    select
        row_number() over (order by occupancy_status) as occupancy_status_id,
        occupancy_status
    from occupancy_statuss
;