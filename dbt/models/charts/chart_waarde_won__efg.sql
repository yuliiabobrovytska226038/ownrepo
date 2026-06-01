with base_rows as (
    select
        db.corporatie,
        db.classificatie,
        v."Energieprestatie (EP2)" as ep2
    from {{ ref("int_chart_woningen_egw_mgw") }} as db
    join {{ source("tms", "vastgoedgegevens_vhe_gegevens") }} as v
      on v."VHE-nummer" = db."VHE-nr"
    where db.corporatie not in ('Huis & Hof', 'Samenwerking Slikkerveer')
),

aggregated as (
    select
        case when grouping(classificatie) = 1 then 'Totaal' else classificatie end as classificatie,
        corporatie,
        count(*) as total_rows,
        count(*) filter (where ep2 is not null and not isnan(ep2)) as total_valid_rows,
        count(*) filter (where ep2 > 290) as count_E_F_G
    from base_rows
    group by grouping sets ((corporatie, classificatie), (corporatie))
)

select
    classificatie as "Classificatie",
    corporatie as "Corporatie",
    total_rows,
    total_valid_rows,
    count_E_F_G,
    count_E_F_G::double / nullif(total_rows, 0) as pct_EFG
from aggregated
order by classificatie, corporatie
