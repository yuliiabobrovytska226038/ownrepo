{% set chart_jaar = var("jaar") %}

with current_year as (
    select
        {{ chart_jaar }} as jaar,
        scenario_types.waarderingstype,
        count(*) filter (where scenario_types.scenario = 'SELLING')::double / nullif(count(*), 0) as percentage
    from {{ ref("dataset_basis") }}
    cross join lateral (
        values
            ('Basis', "Scenario basis"),
            ('Full', scenario_mb)
    ) as scenario_types(waarderingstype, scenario)
    where jaar = {{ chart_jaar }}
      and waarderingsmodel = 'Woningen'
      and scenario_types.scenario is not null
    group by scenario_types.waarderingstype
),

history(jaar, waarderingstype, percentage) as (
    values
        (2021, 'Basis', 0.887),
        (2022, 'Basis', 0.237),
        (2023, 'Basis', 0.712),
        (2024, 'Basis', 0.888),
        (2021, 'Full', 0.780),
        (2022, 'Full', 0.749),
        (2023, 'Full', 0.826),
        (2024, 'Full', 0.854)
)

select jaar as "Jaar", waarderingstype as "Waarderingstype", percentage as "Percentage"
from history
union all
select jaar as "Jaar", waarderingstype as "Waarderingstype", percentage as "Percentage"
from current_year
