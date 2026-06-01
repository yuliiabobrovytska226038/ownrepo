{% set chart_jaar = var("jaar") %}

with basis as (
    select
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
    where "Jaar" = {{ chart_jaar }}
      and "Waarderingsmodel" = 'Woningen'
      and scenario_types.scenario is not null
),

per_type_per_class as (
    select
        waarderingstype as "Waarderingstype",
        "Classificatie",
        "COROP-plusgebieden Code",
        count(*) filter (where scenario = 'SELLING')::double / nullif(count(*), 0) as "Percentage uitponden"
    from basis
    group by waarderingstype, "Classificatie", "COROP-plusgebieden Code"
),

per_type_totaal as (
    select
        waarderingstype as "Waarderingstype",
        'Totaal' as "Classificatie",
        "COROP-plusgebieden Code",
        count(*) filter (where scenario = 'SELLING')::double / nullif(count(*), 0) as "Percentage uitponden"
    from basis
    group by waarderingstype, "COROP-plusgebieden Code"
)

select * from per_type_per_class
union all
select * from per_type_totaal
