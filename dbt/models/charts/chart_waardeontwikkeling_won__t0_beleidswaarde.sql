{#
  Current-year waardeontwikkeling t0 for Beleidswaarde charts.
  Combines: current_t0 marktwaarde + current_t0 beleidswaarde (no historical/CBS).
  Replaces Python calc_waardeontwikkeling_t0(..., "Beleidswaarde").
#}
select "Waarderingstype", "Jaar", "% Mutatie", "Index Mutatie", "Text"
from {{ ref("chart_waardeontwikkeling_won__current_t0") }}
