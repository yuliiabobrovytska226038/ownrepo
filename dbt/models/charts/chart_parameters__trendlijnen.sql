{% set chart_jaar = var("jaar") %}
{% set prior_year = chart_jaar - 1 %}

-- Current year medians per Waarderingstype
with current_medians as (
    select
        {{ chart_jaar }} as "Jaar",
        "Waarderingstype",
        median("LW waarde") as "Median LW",
        median("MH waarde") as "Median Markthuur",
        median("EY doorexploiteren") as "Median EY",
        median("DV doorexploiteren") as "Median DV"
    from {{ ref("dataset_basis") }}
    where "Jaar" = {{ chart_jaar }}
      and "Waarderingsmodel" = 'Woningen'
    group by "Waarderingstype"
),

-- Prior year medians (from matched VHEs in dataset_ontwikkeling)
prior_medians as (
    select
        {{ prior_year }} as "Jaar",
        "Waarderingstype_{{ prior_year }}" as "Waarderingstype",
        median("LW waarde_{{ prior_year }}") as "Median LW",
        median("MH waarde_{{ prior_year }}") as "Median Markthuur",
        median("EY doorexploiteren_{{ prior_year }}") as "Median EY",
        median("DV doorexploiteren_{{ prior_year }}") as "Median DV"
    from {{ ref("dataset_ontwikkeling") }}
    where "Waarderingsmodel" = 'Woningen'
      and _merge = 'both'
    group by "Waarderingstype_{{ prior_year }}"
)

select "Jaar", "Waarderingstype", "Median LW", "Median Markthuur", "Median EY", "Median DV"
from current_medians
union all
select "Jaar", "Waarderingstype", "Median LW", "Median Markthuur", "Median EY", "Median DV"
from prior_medians
order by "Jaar", "Waarderingstype"
