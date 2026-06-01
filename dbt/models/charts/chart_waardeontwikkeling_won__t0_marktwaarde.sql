{#
  Current-year waardeontwikkeling t0 for Marktwaarde charts.
  Combines: current_t0 (excl Beleidswaarde) + historical (current year, only types not in current_t0) + CBS (current year).
  Replaces Python calc_waardeontwikkeling_t0(..., "Marktwaarde").
#}
{% set chart_jaar = var("jaar") %}

select "Waarderingstype", "Jaar", "% Mutatie", "Index Mutatie", "Text"
from {{ ref("chart_waardeontwikkeling_won__current_t0") }}
where "Waarderingstype" != 'Beleidswaarde'

union all

select "Waarderingstype", "Jaar", "% Mutatie", "Index Mutatie",
    {{ format_percent_text('"% Mutatie"') }} as "Text"
from {{ ref("chart_waardeontwikkeling_won__historical") }}
where "Jaar" = {{ chart_jaar }}
  and "Waarderingstype" not in ('Basis', 'Full', 'Beleidswaarde', 'Koopprijsontwikkeling (CBS)')

union all

select
    'Koopprijsontwikkeling (CBS)' as "Waarderingstype",
    "Jaar",
    "mutatie_yoy" as "% Mutatie",
    {{ index_mutatie('"mutatie_yoy"') }} as "Index Mutatie",
    {{ format_percent_text('"mutatie_yoy"') }} as "Text"
from {{ ref("chart_waardeontwikkeling_won__cbs") }}
where "Jaar" = {{ chart_jaar }}
