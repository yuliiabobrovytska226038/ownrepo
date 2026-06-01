{% set chart_jaar = var("jaar") %}

with waarde_basis as (
    select
        jaar,
        {{ normalize_waarderingsmodel("waarderingsmodel") }} as waarderingsmodel,
        handboektype,
        "% Full"
    from {{ ref("dataset_basis") }}
    where jaar = {{ chart_jaar }}
      and waarderingsmodel <> 'Benadering'
      and "Marktwaarde" is not null
),

grouped as (
    select
        b.jaar,
        b.waarderingsmodel,
        b.handboektype,
        avg(b."% Full") as "% Full"
    from waarde_basis as b
    where {{ waarderingsmodel_order("b.waarderingsmodel") }} is not null
      and {{ handboektype_order("b.handboektype") }} is not null
    group by b.jaar, b.waarderingsmodel, b.handboektype
)

select
    'Marktwaarde' as "Waarde",
    jaar as "Jaar",
    waarderingsmodel as "Waarderingsmodel",
    handboektype as "Handboektype",
    "% Full",
    {{ format_percent_text('"% Full"') }} as "Text"
from grouped
order by {{ waarderingsmodel_order("waarderingsmodel") }}, {{ handboektype_order("handboektype") }}, jaar
