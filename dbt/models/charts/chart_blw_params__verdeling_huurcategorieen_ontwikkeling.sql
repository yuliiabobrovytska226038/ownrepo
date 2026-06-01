{% set chart_jaar = var("jaar") %}
{% set prior_year = chart_jaar - 1 %}

with current_year as (
    select
        "Segment huurregime",
        count(*) as "Aantal VHE"
    from {{ source("tms", "policy_value_parameters") }}
    where "Jaar" = {{ chart_jaar }}
    group by "Segment huurregime"
),

prior_year as (
    select
        "Segment huurregime",
        count(*) as "Aantal VHE"
    from {{ source("tms", "policy_value_parameters") }}
    where "Jaar" = {{ prior_year }}
    group by "Segment huurregime"
),

current_totals as (
    select
        "Segment huurregime",
        "Aantal VHE",
        "Aantal VHE" * 1.0 / sum("Aantal VHE") over () as "Percentage"
    from current_year
),

prior_totals as (
    select
        "Segment huurregime",
        "Aantal VHE",
        "Aantal VHE" * 1.0 / sum("Aantal VHE") over () as "Percentage"
    from prior_year
)

select
    coalesce(c."Segment huurregime", p."Segment huurregime") as "Segment huurregime",
    c."Aantal VHE" as "Aantal VHE huidig",
    c."Percentage" as "Percentage huidig",
    p."Aantal VHE" as "Aantal VHE vorig jaar",
    p."Percentage" as "Percentage vorig jaar",
    c."Percentage" - p."Percentage" as "Ontwikkeling percentage"
from current_totals c
full outer join prior_totals p
    on c."Segment huurregime" = p."Segment huurregime"
order by coalesce(c."Segment huurregime", p."Segment huurregime")
