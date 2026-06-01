-- dataset_basis: Central fact table combining all TMS data sources
-- Recreates legacy dataset_basis from duck_query.py
-- 100+ columns: valuation, policy value, parameters, geographic data

{{ config(
    materialized='table',
    post_hook="CREATE INDEX IF NOT EXISTS idx_basis_corp_jaar ON {{ this }} (Corporatie, Jaar)"
) }}

{% set years_1_10 = range(1, 11) %}
{% set years_11_15 = range(11, 16) %}
{% set years_1_15 = range(1, 16) %}
{% set availability_columns = [
    '"Beschikbaarheid (scenario)"',
    '"Beschikbaarheid (eindwaarde)"',
    '"Beschikbaarheid (overdrachtskosten)"',
    '"Beschikbaarheid (60 jaar)"',
] %}

WITH market_value_basis_woningen AS (
    SELECT
        mb.*,
        rum."VHE-nr" AS mapped_vhe_nr
    FROM {{ source('tms', 'market_value_basis') }} mb
    INNER JOIN {{ source('tms', 'rental_unit_mapping') }} rum
        ON mb."rentalUnitInternalId" = rum."rentalUnitInternalId"
        AND mb.Corporatie = rum.Corporatie
        AND mb.Jaar = rum.Jaar
    INNER JOIN {{ source('tms', 'vastgoedgegevens_waarderingscomplexen') }} c
        ON rum.Complexcode = c.Complexcode
        AND rum.Corporatie = c.Corporatie
        AND rum.Jaar = c.Jaar
        -- Legacy Marktwaarde_Basis Oracle SQL filters cplx.valuation_model = 'RESIDENTIAL'.
        AND c.Waarderingsmodel = 'Woningen'
),

merged AS (
SELECT
    w."VHE-nr",
    v.Waarderingscomplex,
    v.Corporatie,
    c.Werkmaatschappij,
    v.Vastgoedcategorie,
    v.Classificatie,
    c.Waarderingsmodel,
    v.Handboektype,
    CAST(v.Jaar AS SMALLINT) AS Jaar,
    v.Peildatum,
    v."Netto huur",
    v."WOZ-waarde primo jaar-1" AS "WOZ-waarde",
    w."Netto marktwaarde" AS Marktwaarde,
    b.Beleidswaarde,
    {% for column_name in availability_columns %}
    COALESCE(b.{{ column_name }}, 0) AS {{ column_name }},
    {% endfor %}
    (
        {% for column_name in availability_columns %}
        COALESCE(b.{{ column_name }}, 0){{ " +" if not loop.last else "" }}
        {% endfor %}
    ) AS Beschikbaarheid_totaal,
    b."Betaalbaarheid",
    COALESCE(b."Kwaliteit (onderhoud)", 0) AS "Kwaliteit (onderhoud)",
    COALESCE(b."Kwaliteit (EFG-labels)", 0) AS "Kwaliteit (EFG-labels)",
    COALESCE(b."Kwaliteit (onderhoud)", 0) + COALESCE(b."Kwaliteit (EFG-labels)", 0) AS Kwaliteit_basis,
    COALESCE(b."Kwaliteit (onderhoud)", 0) + COALESCE(b."Kwaliteit (EFG-labels)", 0) AS Kwaliteit_totaal,
    b."Beheer",
    b.Disconteringsvoet,
    b.Beleidshuur,
    -- Average of 60 years of beleidsonderhoud (policy maintenance)
    (
        {% for yr in range(1, 61) %}
        COALESCE(bp."Beleidsonderhoud jaar {{ yr }}", b."Beleidsonderhoud (norm)", 0){{ " +" if not loop.last else "" }}
        {% endfor %}
    ) / 60.0 AS Beleidsonderhoud,
    b."Beleidsonderhoud (norm)" AS Beleidsonderhoud_norm,
    bp.Beleidsbeheer AS Beleidsbeheer_bp,
    CASE
        WHEN v.Classificatie = 'DAEB' AND CAST(v.Jaar AS SMALLINT) = {{ var('jaar') }} THEN 0.0417
        WHEN v.Classificatie = 'Niet-DAEB' AND CAST(v.Jaar AS SMALLINT) = {{ var('jaar') }} THEN 0.0470
        ELSE NULL
    END AS "Disconteringsvoet Beleidswaarde",
    v.Postcode,
    'GM' || lpad(CAST(wp.gemeentecode AS VARCHAR), 4, '0') AS Gemeentecode,
    mppf."Aantal vrijheidsgraden toegepast",
    mppf."% Full",
    mppf."Aantal vrije waardering overrules toegepast",
    mppf."% Vrije waardering",
    mppf."Waarderingstype",
    COALESCE(v."Erfpacht waardecorrectie doorexploiteren", 0) AS "Erfpacht waardecorrectie doorexploiteren basis",
    COALESCE(v."Erfpacht waardecorrectie uitponden", 0) AS "Erfpacht waardecorrectie uitponden basis",
    COALESCE(w."Erfpacht", 0) AS "Erfpacht NCW",
    -- market_value_basis joined via rental_unit_mapping (rentalUnitInternalId -> VHE-nr)
    mb."Marktwaarde" AS "Marktwaarde_1",
    mb."Marktwaarde doorexploiteren",
    mb."Marktwaarde uitponden",
    mb."Scenario" AS "Scenario_mb",
    mb."Marktwaarde basis",
    mb."Marktwaarde basis doorexploiteren",
    mb."Marktwaarde basis uitponden",
    mb."Scenario basis",
    vmp."Erfpacht (EP)",
    COALESCE(vmp."EP suppletie bij verkoop", 0) AS "EP suppletie bij verkoop",
    {% for yr in years_1_10 %}
    COALESCE(vmp."EP doorexploiteren jaar {{ yr }}", 0) +
    {% endfor %}
    (
        {% for yr in years_11_15 %}
        COALESCE(vmp."EP doorexploiteren jaar {{ yr }}", 0){{ " +" if not loop.last else "" }}
        {% endfor %}
    ) / 15 AS "EP doorexploiteren jaar 1-15",
    {% for yr in years_1_10 %}
    COALESCE(vmp."EP uitponden jaar {{ yr }}", 0) +
    {% endfor %}
    (
        {% for yr in years_11_15 %}
        COALESCE(vmp."EP uitponden jaar {{ yr }}", 0){{ " +" if not loop.last else "" }}
        {% endfor %}
    ) / 15 AS "EP uitponden jaar 1-15",
    po."EP waardecorrectie doorexploiteren",
    po."EP waardecorrectie uitponden",
    po."Bron disconteringsvoet (DV)",
    po."DV doorexploiteren",
    po."DV doorexploiteren handboek",
    po."DV uitponden",
    po."DV uitponden handboek",
    po."Bron leegwaarde (LW)",
    po."LW waarde",
    po."LW handboek waarde",
    po."Bron markthuur (MH)",
    po."MH waarde",
    po."MH handboek waarde",
    po."Bron Onderhoud (OH)",
    po."OH doorexploiteren",
    po."OH doorexploiteren handboek",
    po."OH uitponden",
    po."OH uitponden handboek",
    po."Bron mutatiegraad doorexploiteren (MD)",
    po."MD waarde",
    po."MD handboek waarde",
    po."Bron mutatiegraad uitponden (MU)",
    (
        {% for yr in years_1_15 %}
        po."MU jaar {{ yr }}"{{ " +" if not loop.last else "" }}
        {% endfor %}
    ) / 15 AS "MU jaar 1-15",
    (
        {% for yr in years_1_15 %}
        po."MU jaar {{ yr }} handboek"{{ " +" if not loop.last else "" }}
        {% endfor %}
    ) / 15 AS "MU jaar 1-15 handboek",
    po."Bron exit yield (EY)",
    po."EY doorexploiteren",
    po."EY doorexploiteren handboek",
    po."EY uitponden",
    po."EY uitponden handboek",
    po."Bron scenario (SC)",
    po."SC waarde",
    po."Bron mutatieonderhoud (MOH)",
    po."Bron markthuurstijging (MHS)",
    po."Bron leegwaardestijging verleden (LSV)",
    po."Bron leegwaardestijging (LS)",
    po."Bron mutatiegraad doorexploiteren (MD)",
    po."Bron mutatiegraad uitponden (MU)",
    po."Bron overige kosten (OVK)",
    po."Bron overige opbrengsten (OVO)",
    po."Bron technische splitsingskosten (TS)",
    po."Bron gedeelte niet verkopen bij mutatie (VM)",
    wp.gemeente AS Gemeente,
    wp.corop_gebied AS "COROP-gebied",
    wp.regio AS Regio,
    wp.categorie AS "Krimp/aardbeving",
    cg.corop_code AS "COROP-plusgebieden Code",
    cg.corop_naam AS "COROP-plusgebieden Naam",
    cg.provincie_code AS "Provincies Code",
    cg.provincie_naam AS "Provincies Naam"
FROM {{ source('tms', 'vastgoedgegevens_vhe_gegevens') }} v
FULL OUTER JOIN {{ source('tms', 'vastgoedgegevens_waarderingscomplexen') }} c
    ON v.Waarderingscomplex = c.Complexcode
    AND v.Corporatie = c.Corporatie
    AND v.Peildatum = c.Peildatum
FULL OUTER JOIN {{ source('tms', 'valuation_overview_vhe') }} w
    ON v."VHE-nummer" = w."VHE-nr"
    AND v.Corporatie = w.Corporatie
    AND v.Peildatum = w.Peildatum
FULL OUTER JOIN {{ source('tms', 'policy_value_report_vhe') }} b
    ON w."VHE-nr" = b."VHE-nr"
    AND w.Corporatie = b.Corporatie
    AND w.Peildatum = b.Peildatum
FULL OUTER JOIN {{ ref('int_marktwaardeparameters') }} vmp
    ON v."VHE-nummer" = vmp."VHE-nr"
    AND v.Corporatie = vmp.Corporatie
    AND v.Peildatum = vmp.Peildatum
FULL OUTER JOIN {{ ref('stg_tms_percentage_full') }} mppf
    ON v."VHE-nummer" = mppf."VHE-nr"
    AND v.Corporatie = mppf."Corporatie"
    AND v.Peildatum = mppf."Peildatum"
FULL OUTER JOIN {{ ref('int_parameteroverzicht') }} po
    ON v."VHE-nummer" = po."VHE-nr"
    AND v.Corporatie = po.Corporatie
    AND v.Peildatum = po.Peildatum
LEFT JOIN {{ source('tms', 'policy_value_parameters') }} bp
    ON bp."VHE-nr" = v."VHE-nummer"
    AND bp.Corporatie = v.Corporatie
    AND bp.Peildatum = v.Peildatum
LEFT JOIN market_value_basis_woningen mb
    ON mb.mapped_vhe_nr = v."VHE-nummer"
    AND mb.Corporatie = v.Corporatie
    AND mb.Jaar = v.Jaar
LEFT JOIN {{ source('external', 'postal_code_mapping') }} wp
    ON v.Postcode[1:4] = wp.postcode
LEFT JOIN {{ source('external', 'cbs_corop_regions') }} cg
    ON 'GM' || lpad(CAST(wp.gemeentecode AS VARCHAR), 4, '0') = cg.gemeentecode
)
SELECT * FROM merged
