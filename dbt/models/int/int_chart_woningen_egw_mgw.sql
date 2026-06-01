{% set chart_jaar = var("jaar") %}

select *
from {{ ref("dataset_basis") }}
where jaar = {{ chart_jaar }}
  and handboektype in ('EGW', 'MGW')
  and waarderingsmodel = 'Woningen'
