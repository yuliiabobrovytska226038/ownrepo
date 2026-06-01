{% set prior_year = var("jaar") - 1 %}

with {{ waardeontwikkeling_current_base_ctes() }},

grouped as (
    select
        waarderingsmodel,
        handboektype,
        avg(marktwaarde) as marktwaarde,
        avg(marktwaarde_prior) as "Marktwaarde_{{ prior_year }}",
        count("VHE-nr") as "Aantal VHE"
    from waardeontwikkeling_current_base
    where handboektype = handboektype_prior
    group by waarderingsmodel, handboektype
),

metrics as (
    select
        waarderingsmodel,
        handboektype,
        marktwaarde,
        "Marktwaarde_{{ prior_year }}",
        "Aantal VHE",
        {{ mutation_columns_when_current_nonzero("marktwaarde", '"Marktwaarde_' ~ prior_year ~ '"') }}
    from grouped
)

select
    waarderingsmodel as "Waarderingsmodel",
    handboektype as "Handboektype",
    marktwaarde as "Marktwaarde",
    "Marktwaarde_{{ prior_year }}",
    "Aantal VHE",
    euro_mutatie as "€ Mutatie",
    pct_mutatie as "% Mutatie",
    {{ index_mutatie("pct_mutatie") }} as "Index Mutatie",
    {{ format_percent_text("pct_mutatie") }} as "Text"
from metrics
order by
    {{ waarderingsmodel_order("waarderingsmodel") }},
    {{ handboektype_order("handboektype") }}
