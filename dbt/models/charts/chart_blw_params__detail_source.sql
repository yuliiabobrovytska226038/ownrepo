{% set chart_jaar = var("jaar") %}
{% set prior_year = chart_jaar - 1 %}

-- VHE-level detail for beleidswaarde parameter COROP maps (items 10-12)
-- Provides per-VHE beleidswaarde parameters for both years, used in Excel "Detail" sheet
select
    "Corporatie",
    "VHE-nr",
    "Waarderingscomplex",
    "COROP-plusgebieden Code",
    "COROP-plusgebieden Naam",
    "Handboektype",
    "Classificatie",
    "Waarderingstype",
    TRY_CAST("Beleidsonderhoud" AS DOUBLE) as "Beleidsonderhoud",
    TRY_CAST("Beleidsonderhoud_{{ prior_year }}" AS DOUBLE) as "Beleidsonderhoud vorig jaar",
    TRY_CAST("Beleidsbeheer_bp" AS DOUBLE) as "Beleidsbeheer",
    TRY_CAST("Beleidsbeheer_{{ prior_year }}" AS DOUBLE) as "Beleidsbeheer vorig jaar",
    TRY_CAST("Beleidshuur" AS DOUBLE) as "Beleidshuur",
    TRY_CAST("Beleidshuur_{{ prior_year }}" AS DOUBLE) as "Beleidshuur vorig jaar",
    TRY_CAST("Beleidswaarde" AS DOUBLE) as "Beleidswaarde",
    TRY_CAST("Beleidswaarde_{{ prior_year }}" AS DOUBLE) as "Beleidswaarde vorig jaar",
    TRY_CAST("Marktwaarde" AS DOUBLE) as "Marktwaarde",
    TRY_CAST("Netto huur" AS DOUBLE) as "Netto huur"
from {{ ref("dataset_ontwikkeling") }}
where "Waarderingsmodel" = 'Woningen'
  and _merge = 'both'
