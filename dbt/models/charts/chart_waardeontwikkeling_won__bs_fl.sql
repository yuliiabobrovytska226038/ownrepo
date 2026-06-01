{% set prior_year = var("jaar") - 1 %}

select
    case
        when "WaardeFilter" = 'Beleidswaarde_uitgesplitst' then 'Beleidswaarde'
        else "WaardeFilter"
    end as "Waarde",
    "Waarderingstype",
    "Jaar",
    "Marktwaarde",
    "Marktwaarde_{{ prior_year }}",
    "Beleidswaarde",
    "Beleidswaarde_{{ prior_year }}",
    "Aantal VHE",
    "Waardeontwikkeling",
    "€ Mutatie",
    "% Mutatie",
    "Index Mutatie",
    "Text"
from {{ ref("chart_waardeontwikkeling_won__waarde") }}
where "WaardeFilter" in ('Marktwaarde', 'Beleidswaarde_uitgesplitst')
order by "Waarde", "Waarderingstype", "Jaar"
