{% set chart_jaar = var("jaar") %}

with waarde_basis as (
    select
        jaar,
        {{ normalize_waarderingsmodel("waarderingsmodel") }} as waarderingsmodel,
        waarde,
        waarde_bedrag
    from {{ ref("dataset_basis") }}
    cross join lateral {{ waarde_types_lateral() }}
    where jaar = {{ chart_jaar }}
      and waarderingsmodel <> 'Benadering'
      and waarde_bedrag is not null
),

grouped as (
    select
        waarde,
        jaar,
        waarderingsmodel,
        round(sum(waarde_bedrag) / nullif(sum(sum(waarde_bedrag)) over (partition by waarde, jaar), 0), 3) as "% van totaal"
    from waarde_basis
    group by waarde, jaar, waarderingsmodel
)

select
    waarde as "Waarde",
    jaar as "Jaar",
    waarderingsmodel as "Waarderingsmodel",
    "% van totaal",
    {{ format_percent_text('"% van totaal"') }} as "Text"
from grouped
