{% set chart_jaar = var("jaar") %}
{% set prior_year = chart_jaar - 1 %}

with basis as (
    select
        "Jaar",
        "COROP-plusgebieden Code",
        "Classificatie",
        scenario_types.waarderingstype,
        scenario_types.scenario
    from {{ ref("dataset_basis") }}
    cross join lateral (
        values
            ('Basis', "Scenario basis"),
            ('Full', "Scenario_mb")
    ) as scenario_types(waarderingstype, scenario)
    where "Jaar" in ({{ chart_jaar }}, {{ prior_year }})
      and "Waarderingsmodel" = 'Woningen'
      and scenario_types.scenario is not null
),

per_year_per_class as (
    select
        "Jaar",
        waarderingstype as "Waarderingstype",
        "Classificatie",
        "COROP-plusgebieden Code",
        count(*) filter (where scenario = 'SELLING')::double / nullif(count(*), 0) as pct_uitponden
    from basis
    group by "Jaar", waarderingstype, "Classificatie", "COROP-plusgebieden Code"
),

per_year_totaal as (
    select
        "Jaar",
        waarderingstype as "Waarderingstype",
        'Totaal' as "Classificatie",
        "COROP-plusgebieden Code",
        count(*) filter (where scenario = 'SELLING')::double / nullif(count(*), 0) as pct_uitponden
    from basis
    group by "Jaar", waarderingstype, "COROP-plusgebieden Code"
),

per_year as (
    select * from per_year_per_class
    union all
    select * from per_year_totaal
),

current_pct as (
    select * from per_year where "Jaar" = {{ chart_jaar }}
),

prior_pct as (
    select * from per_year where "Jaar" = {{ prior_year }}
)

select
    c."Waarderingstype",
    c."Classificatie",
    c."COROP-plusgebieden Code",
    c.pct_uitponden as "Percentage uitponden",
    p.pct_uitponden as "Percentage uitponden vorig jaar",
    c.pct_uitponden - p.pct_uitponden as "Ontwikkeling uitponden"
from current_pct c
join prior_pct p on c."Waarderingstype" = p."Waarderingstype"
    and c."Classificatie" = p."Classificatie"
    and c."COROP-plusgebieden Code" = p."COROP-plusgebieden Code"
