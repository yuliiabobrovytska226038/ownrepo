{% set chart_jaar = var("jaar") %}
{% set prior_year = chart_jaar - 1 %}

with basis as (
    select
        "COROP-plusgebieden Code",
        "Classificatie",
        TRY_CAST("Beleidsonderhoud" AS DOUBLE) as bo_current,
        TRY_CAST("Beleidsonderhoud_{{ prior_year }}" AS DOUBLE) as bo_prior
    from {{ ref("dataset_ontwikkeling") }}
    where "Waarderingsmodel" = 'Woningen'
      and _merge = 'both'
      and "Beleidsonderhoud" is not null
      and "Beleidsonderhoud_{{ prior_year }}" is not null
),

per_classificatie as (
    select
        "Classificatie",
        "COROP-plusgebieden Code",
        median(bo_current) as "Median beleidsonderhoud",
        median(bo_prior) as "Median beleidsonderhoud vorig jaar",
        median(bo_current) - median(bo_prior) as "Ontwikkeling beleidsonderhoud"
    from basis
    group by "Classificatie", "COROP-plusgebieden Code"
),

totaal as (
    select
        'Totaal' as "Classificatie",
        "COROP-plusgebieden Code",
        median(bo_current) as "Median beleidsonderhoud",
        median(bo_prior) as "Median beleidsonderhoud vorig jaar",
        median(bo_current) - median(bo_prior) as "Ontwikkeling beleidsonderhoud"
    from basis
    group by "COROP-plusgebieden Code"
)

select * from per_classificatie
union all
select * from totaal
