{% set prior_year = var("jaar") - 1 %}

select
    "Waarderingstype",
    "Jaar",
    "Marktwaarde",
    "Marktwaarde_{{ prior_year }}",
    "Aantal VHE",
    "Waardeontwikkeling",
    "€ Mutatie",
    "% Mutatie",
    "Index Mutatie",
    "Text",
    "Beleidswaarde",
    "Beleidswaarde_{{ prior_year }}"
from {{ ref("chart_waardeontwikkeling_won__waarde") }}
where "WaardeFilter" in ('Marktwaarde', 'Beleidswaarde')
order by "Waarderingstype", "Jaar"
