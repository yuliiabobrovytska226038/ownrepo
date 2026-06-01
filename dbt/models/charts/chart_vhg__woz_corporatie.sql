{% set chart_jaar = var("jaar") %}
{% set prior_year = var("jaar") - 1 %}

with woz_by_year as (
    select
        jaar,
        corporatie,
        median("WOZ-waarde") as median_woz
    from {{ ref("dataset_basis") }}
    where jaar in ({{ chart_jaar }}, {{ prior_year }})
      and handboektype in ('EGW', 'MGW')
      and waarderingsmodel = 'Woningen'
    group by jaar, corporatie
)

select
    w_current.corporatie as "Corporatie",
    w_current.median_woz as "median_woz_{{ chart_jaar }}",
    w_prior.median_woz as "median_woz_{{ prior_year }}",
    round((w_current.median_woz - w_prior.median_woz) / nullif(w_prior.median_woz, 0), 4) as "Stijging WOZ-waarde"
from woz_by_year as w_current
join woz_by_year as w_prior
  on w_current.corporatie = w_prior.corporatie
where w_current.jaar = {{ chart_jaar }}
  and w_prior.jaar = {{ prior_year }}
order by w_current.corporatie
