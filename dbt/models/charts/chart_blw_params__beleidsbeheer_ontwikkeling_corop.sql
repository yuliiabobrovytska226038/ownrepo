{% set chart_jaar = var("jaar") %}
{% set prior_year = chart_jaar - 1 %}

with basis as (
    select
        "COROP-plusgebieden Code",
        "Classificatie",
        TRY_CAST("Beleidsbeheer_bp" AS DOUBLE) as bb_current,
        TRY_CAST("Beleidsbeheer_{{ prior_year }}" AS DOUBLE) as bb_prior
    from {{ ref("dataset_ontwikkeling") }}
    where "Waarderingsmodel" = 'Woningen'
      and _merge = 'both'
      and "Beleidsbeheer_bp" is not null
      and "Beleidsbeheer_{{ prior_year }}" is not null
),

per_classificatie as (
    select
        "Classificatie",
        "COROP-plusgebieden Code",
        median(bb_current) as "Median beleidsbeheer",
        median(bb_prior) as "Median beleidsbeheer vorig jaar",
        median(bb_current) - median(bb_prior) as "Ontwikkeling beleidsbeheer"
    from basis
    group by "Classificatie", "COROP-plusgebieden Code"
),

totaal as (
    select
        'Totaal' as "Classificatie",
        "COROP-plusgebieden Code",
        median(bb_current) as "Median beleidsbeheer",
        median(bb_prior) as "Median beleidsbeheer vorig jaar",
        median(bb_current) - median(bb_prior) as "Ontwikkeling beleidsbeheer"
    from basis
    group by "COROP-plusgebieden Code"
)

select * from per_classificatie
union all
select * from totaal
