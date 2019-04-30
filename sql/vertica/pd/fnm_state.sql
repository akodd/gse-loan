drop table fnm_state;
create table fnm_state as
    with states as (
        select
            distinct state
        from fnm_acq
        where state is not null
    )
    select
        row_number() over (order by state) as state_id,
        state
    from states
;