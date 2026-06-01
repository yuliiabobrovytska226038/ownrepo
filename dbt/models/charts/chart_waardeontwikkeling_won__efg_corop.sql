{% set chart_jaar = var("jaar") %}

with base as (
    select
        db."COROP-plusgebieden Code",
        db."Classificatie",
        v."Energieprestatie (EP2)"
    from {{ ref("dataset_basis") }} as db
    join {{ source("tms", "vastgoedgegevens_vhe_gegevens") }} as v
      on v."VHE-nummer" = db."VHE-nr"
    where db.jaar = {{ chart_jaar }}
      and db.handboektype in ('EGW', 'MGW')
),

per_classificatie as (
    select
        "Classificatie",
        "COROP-plusgebieden Code",
        count(*) as total_rows,
        count(*) filter (where "Energieprestatie (EP2)" > 290) as count_E_F_G,
        count(*) filter (where "Energieprestatie (EP2)" > 290)::double / nullif(count(*), 0) as perc_EFG
    from base
    group by "Classificatie", "COROP-plusgebieden Code"
),

totaal as (
    select
        'Totaal' as "Classificatie",
        "COROP-plusgebieden Code",
        count(*) as total_rows,
        count(*) filter (where "Energieprestatie (EP2)" > 290) as count_E_F_G,
        count(*) filter (where "Energieprestatie (EP2)" > 290)::double / nullif(count(*), 0) as perc_EFG
    from base
    group by "COROP-plusgebieden Code"
)

select * from per_classificatie
union all
select * from totaal
