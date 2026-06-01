{% set chart_jaar = var("jaar") %}

select *
from {{ source("tms", "tms_verschillenanalyse_marketvalue") }}
where jaar = {{ chart_jaar }}
