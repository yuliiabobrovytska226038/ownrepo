{% set prior_year = var("jaar") - 1 %}

select
    *,
    beleidsbeheer_bp as "Beleidsbeheer",
    "Beleidsbeheer_{{ prior_year }}" as "Beleidsbeheer_{{ prior_year }}"
from {{ ref("int_chart_ontwikkeling_woningen_egw_mgw") }}
where beleidswaarde is not null
