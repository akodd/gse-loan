drop table fnm_msa;
create table fnm_msa as
    with msas as (
        select
            distinct msa
        from fnm_prf
        where msa is not null
    )
    select
        row_number() over (order by msa) as msa_id,
        msa
    from msas
;
