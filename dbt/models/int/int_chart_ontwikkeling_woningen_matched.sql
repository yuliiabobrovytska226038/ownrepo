{% set chart_jaar = var("jaar") %}
{% set prior_year = var("jaar") - 1 %}

select *
from {{ ref("dataset_ontwikkeling") }}
where waarderingsmodel = "Waarderingsmodel_{{ prior_year }}"
  and waarderingsmodel = 'Woningen'
  and _merge = 'both'
