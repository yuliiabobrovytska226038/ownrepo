{% set chart_jaar = var("jaar") %}

with aggregated as (
    select
        classificatie as "Classificatie",
        avg("HS jaar 1") as hs_year_1,
        avg("HS jaar 2") as hs_year_2,
        avg("HS jaar 3") as hs_year_3,
        avg("HS jaar 4") as hs_year_4,
        avg("HS jaar 5") as hs_year_5,
        avg("HS jaar 6 e.v.") as hs_year_6
    from {{ source("tms", "policy_value_parameters") }}
    where jaar = {{ chart_jaar }}
      and waarderingsmodel = 'Woningen'
    group by classificatie
)

select
    "Classificatie",
    step_jaar::integer as "Jaar",
    huurstijging as "Huurstijging"
from aggregated
cross join lateral (
    values
        (1, hs_year_1),
        (2, hs_year_2),
        (3, hs_year_3),
        (4, hs_year_4),
        (5, hs_year_5),
        (6, hs_year_6)
) as steps(step_jaar, huurstijging)

order by "Classificatie", "Jaar"
