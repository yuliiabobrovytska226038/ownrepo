select
    valuation_model as "Waarderingsmodel",
    pct_mutatie as "% Mutatie",
    jaar as "Jaar",
    {{ format_percent_text("pct_mutatie") }} as "Text"
from {{ source("external", "historical_data") }}
where sheet = 'Waardeontwikkeling per model'
