{% set chart_jaar = var("jaar") %}
{% set prior_year = chart_jaar - 1 %}

with basis as (
    select
        "COROP-plusgebieden Code",
        "Classificatie",
        TRY_CAST("Beleidshuur" AS DOUBLE) as bh_current,
        TRY_CAST("Beleidshuur_{{ prior_year }}" AS DOUBLE) as bh_prior
    from {{ ref("dataset_ontwikkeling") }}
    where "Waarderingsmodel" = 'Woningen'
      and _merge = 'both'
      and "Beleidshuur" is not null
      and "Beleidshuur_{{ prior_year }}" is not null
),

per_classificatie as (
    select
        "Classificatie",
        "COROP-plusgebieden Code",
        median(bh_current) as "Median beleidshuur",
        median(bh_prior) as "Median beleidshuur vorig jaar",
        median(bh_current) - median(bh_prior) as "Ontwikkeling beleidshuur"
    from basis
    group by "Classificatie", "COROP-plusgebieden Code"
),

totaal as (
    select
        'Totaal' as "Classificatie",
        "COROP-plusgebieden Code",
        median(bh_current) as "Median beleidshuur",
        median(bh_prior) as "Median beleidshuur vorig jaar",
        median(bh_current) - median(bh_prior) as "Ontwikkeling beleidshuur"
    from basis
    group by "COROP-plusgebieden Code"
)

select * from per_classificatie
union all
select * from totaal
