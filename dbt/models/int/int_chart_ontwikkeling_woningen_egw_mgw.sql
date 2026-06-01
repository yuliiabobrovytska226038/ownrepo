{% set chart_jaar = var("jaar") %}
{% set prior_year = var("jaar") - 1 %}

select
    *,
    case
        when gemeente in ('Rotterdam', '''s-Gravenhage', 'Amsterdam', 'Utrecht-(gemeente)')
            then replace(gemeente, 'Utrecht-(gemeente)', 'Utrecht-stad')
        else "Provincies Naam"
    end as "Indexatiegebied"
from {{ ref("dataset_ontwikkeling") }}
where waarderingsmodel = "Waarderingsmodel_{{ prior_year }}"
  and waarderingsmodel = 'Woningen'
  and handboektype in ('EGW', 'MGW')
  and "Handboektype_{{ prior_year }}" in ('EGW', 'MGW')
