{% set chart_jaar = var("jaar") %}

with basis as (
    select
        "Segment huurregime",
        count(*) as "Aantal VHE"
    from {{ source("tms", "policy_value_parameters") }}
    where "Jaar" = {{ chart_jaar }}
    group by "Segment huurregime"
)

select
    "Segment huurregime",
    "Aantal VHE",
    "Aantal VHE" * 1.0 / sum("Aantal VHE") over () as "Percentage"
from basis
order by "Segment huurregime"
