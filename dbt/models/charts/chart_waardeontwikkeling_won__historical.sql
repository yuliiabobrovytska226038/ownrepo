select
    valuation_type as "Waarderingstype",
    jaar::integer as "Jaar",
    pct_mutatie as "% Mutatie",
    index_mutatie as "Index Mutatie",
    indexcijfer as "Indexcijfer"
from {{ source("external", "historical_data") }}
where sheet = 'Waardeontwikkeling'
