-- dataset_ontwikkeling: Year-over-year comparison
-- Recreates legacy dataset_ontwikkeling from duck_query.py
-- Joins current year (var('jaar')) with prior year (var('jaar')-1) by VHE-nr + Corporatie

{{ config(
    materialized='table',
    post_hook="CREATE INDEX IF NOT EXISTS idx_ontw_corp ON {{ this }} (Corporatie)"
) }}

{% set jaar = var('jaar') %}
{% set jaar_m1 = var('jaar') - 1 %}
{% set prior_year_aliases = {
    "Scenario_mb": "Scenario",
    "Beschikbaarheid_totaal": "Beschikbaarheid",
    "Kwaliteit_totaal": "Kwaliteit",
    "Beleidsbeheer_bp": "Beleidsbeheer",
} %}
{% set prior_year_columns = [
    "Waarderingscomplex",
    "Werkmaatschappij",
    "Vastgoedcategorie",
    "Classificatie",
    "Waarderingsmodel",
    "Handboektype",
    "Jaar",
    "Peildatum",
    "Marktwaarde",
    "Scenario_mb",
    "Beleidswaarde",
    "Beschikbaarheid_totaal",
    "Betaalbaarheid",
    "Kwaliteit_totaal",
    "Beheer",
    "Beleidshuur",
    "Beleidsonderhoud",
    "Beleidsbeheer_bp",
    "Postcode",
    "Gemeentecode",
    "Aantal vrijheidsgraden toegepast",
    "% Full",
    "Aantal vrije waardering overrules toegepast",
    "% Vrije waardering",
    "Waarderingstype",
    "Bron disconteringsvoet (DV)",
    "DV doorexploiteren",
    "DV doorexploiteren handboek",
    "DV uitponden",
    "DV uitponden handboek",
    "Bron leegwaarde (LW)",
    "LW waarde",
    "LW handboek waarde",
    "Bron markthuur (MH)",
    "MH waarde",
    "MH handboek waarde",
    "Bron Onderhoud (OH)",
    "OH doorexploiteren",
    "OH doorexploiteren handboek",
    "OH uitponden",
    "OH uitponden handboek",
    "Bron mutatiegraad doorexploiteren (MD)",
    "MD waarde",
    "MD handboek waarde",
    "Bron mutatiegraad uitponden (MU)",
    "MU jaar 1-15",
    "MU jaar 1-15 handboek",
    "Bron exit yield (EY)",
    "EY doorexploiteren",
    "EY doorexploiteren handboek",
    "EY uitponden",
    "EY uitponden handboek",
    "Bron scenario (SC)",
    "SC waarde",
    "Gemeente",
    "COROP-gebied",
    "Regio",
    "Krimp/aardbeving",
    "COROP-plusgebieden Code",
    "COROP-plusgebieden Naam",
    "Provincies Code",
    "Provincies Naam",
] %}

WITH year_0 AS (
    SELECT *
    FROM {{ ref('dataset_basis') }}
    WHERE Jaar = {{ jaar }}
),
year_m1 AS (
    SELECT *
    FROM {{ ref('dataset_basis') }}
    WHERE Jaar = {{ jaar_m1 }}
),
merged AS (
    SELECT
        y0.*,
        {% for source_column in prior_year_columns %}
        ym1."{{ source_column }}" AS "{{ prior_year_aliases.get(source_column, source_column) }}_{{ jaar_m1 }}",
        {% endfor %}
        CASE
            WHEN y0."VHE-nr" IS NOT NULL AND ym1."VHE-nr" IS NOT NULL THEN 'both'
            WHEN y0."VHE-nr" IS NOT NULL THEN 'left_only'
            WHEN ym1."VHE-nr" IS NOT NULL THEN 'right_only'
        END AS _merge
    FROM year_0 y0
    FULL OUTER JOIN year_m1 ym1
        ON y0."VHE-nr" = ym1."VHE-nr"
        AND y0.Corporatie = ym1.Corporatie
)
SELECT *
FROM merged
