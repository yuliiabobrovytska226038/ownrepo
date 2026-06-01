{#
  COROP-level year-over-year development for DV and Beleidsonderhoud.
  Replaces two Python groupby/assign blocks with one SQL model.
  Aggregates at VHE (unit) level per COROP.
  Includes Classificatie dimension (DAEB / Niet-DAEB / Totaal).
#}
{% set prior_year = var("jaar") - 1 %}

with source as (
    select * from {{ ref("chart_waardeontwikkeling_won__ontwikkeling_source") }}
),

per_classificatie as (
    select
        "Classificatie",
        "COROP-plusgebieden Code",
        median(
            "Disconteringsvoet Beleidswaarde"
            - try_cast("DV doorexploiteren_{{ prior_year }}" as double)
        ) as median_dv_delta,
        median("Beleidsonderhoud") as median_beleidsonderhoud,
        median("Beleidsonderhoud_{{ prior_year }}") as median_beleidsonderhoud_prior
    from source
    group by "Classificatie", "COROP-plusgebieden Code"
),

totaal as (
    select
        'Totaal' as "Classificatie",
        "COROP-plusgebieden Code",
        median(
            "Disconteringsvoet Beleidswaarde"
            - try_cast("DV doorexploiteren_{{ prior_year }}" as double)
        ) as median_dv_delta,
        median("Beleidsonderhoud") as median_beleidsonderhoud,
        median("Beleidsonderhoud_{{ prior_year }}") as median_beleidsonderhoud_prior
    from source
    group by "COROP-plusgebieden Code"
),

combined as (
    select * from per_classificatie
    union all
    select * from totaal
)

select
    "Classificatie",
    "COROP-plusgebieden Code",
    median_dv_delta,
    case
        when median_beleidsonderhoud_prior = 0 then null
        else (median_beleidsonderhoud - median_beleidsonderhoud_prior)
             / median_beleidsonderhoud_prior
    end as ontwikkeling_beleidsonderhoud
from combined
