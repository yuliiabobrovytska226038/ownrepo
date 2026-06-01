{% set chart_jaar = var("jaar") %}
{% set prior_year = chart_jaar - 1 %}

with basis as (
    select
        "COROP-plusgebieden Code",
        "Waarderingstype",
        "Classificatie",
        "DV doorexploiteren",
        "DV doorexploiteren_{{ prior_year }}"
    from {{ ref("dataset_ontwikkeling") }}
    where "Waarderingsmodel" = 'Woningen'
      and _merge = 'both'
      and "DV doorexploiteren" is not null
      and "DV doorexploiteren_{{ prior_year }}" is not null
),

per_type_per_class as (
    select
        "Waarderingstype",
        "Classificatie",
        "COROP-plusgebieden Code",
        median("DV doorexploiteren") as "Median DV",
        median("DV doorexploiteren_{{ prior_year }}") as "Median DV vorig jaar",
        median("DV doorexploiteren") - median("DV doorexploiteren_{{ prior_year }}") as "Ontwikkeling DV"
    from basis
    group by "Waarderingstype", "Classificatie", "COROP-plusgebieden Code"
),

per_type_totaal as (
    select
        "Waarderingstype",
        'Totaal' as "Classificatie",
        "COROP-plusgebieden Code",
        median("DV doorexploiteren") as "Median DV",
        median("DV doorexploiteren_{{ prior_year }}") as "Median DV vorig jaar",
        median("DV doorexploiteren") - median("DV doorexploiteren_{{ prior_year }}") as "Ontwikkeling DV"
    from basis
    group by "Waarderingstype", "COROP-plusgebieden Code"
),

totaal_per_class as (
    select
        'Totaal' as "Waarderingstype",
        "Classificatie",
        "COROP-plusgebieden Code",
        median("DV doorexploiteren") as "Median DV",
        median("DV doorexploiteren_{{ prior_year }}") as "Median DV vorig jaar",
        median("DV doorexploiteren") - median("DV doorexploiteren_{{ prior_year }}") as "Ontwikkeling DV"
    from basis
    group by "Classificatie", "COROP-plusgebieden Code"
),

totaal_totaal as (
    select
        'Totaal' as "Waarderingstype",
        'Totaal' as "Classificatie",
        "COROP-plusgebieden Code",
        median("DV doorexploiteren") as "Median DV",
        median("DV doorexploiteren_{{ prior_year }}") as "Median DV vorig jaar",
        median("DV doorexploiteren") - median("DV doorexploiteren_{{ prior_year }}") as "Ontwikkeling DV"
    from basis
    group by "COROP-plusgebieden Code"
)

select * from per_type_per_class
union all
select * from per_type_totaal
union all
select * from totaal_per_class
union all
select * from totaal_totaal
