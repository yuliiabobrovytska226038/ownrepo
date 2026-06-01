-- int_marktwaarde: Market value params + valuation overview
-- Recreates legacy vw_marktwaarde from duck_query.py

WITH marktwaardeparameters AS (
    {{ market_value_vhe_parameter_sources() }}
),
complex_parameters AS (
    {{ market_value_complex_parameter_sources() }}
),
merge AS (
    SELECT
        COALESCE(m.Corporatie, c.Corporatie, w.Corporatie) AS Corporatie,
        COALESCE(m.Jaar, c.Jaar, w.Jaar) AS Jaar,
        COALESCE(m.Peildatum, c.Peildatum, w.Peildatum) AS Peildatum,
        COALESCE(m.Complexcode, c.Complexcode) AS Complexcode,
        COALESCE(m."VHE-nr", w."VHE-nr") AS "VHE-nr",
        m.* EXCLUDE (Corporatie, Jaar, Peildatum, Complexcode, "VHE-nr", Complexnaam, Waarderingsmodel, Deelportefeuille, Classificatie, Adres),
        c.* EXCLUDE (Corporatie, Jaar, Peildatum, Complexcode, Complexnaam, Waarderingsmodel, Deelportefeuille),
        w.* EXCLUDE (Corporatie, Jaar, Peildatum, "VHE-nr", Waarderingscomplex, Complexnaam, Deelportefeuille, Classificatie, Adres, Postcode),
        COALESCE(m.Complexnaam, c.Complexnaam, w.Complexnaam) AS Complexnaam,
        COALESCE(m.Deelportefeuille, c.Deelportefeuille, w.Deelportefeuille) AS Deelportefeuille,
        COALESCE(m.Classificatie, w.Classificatie) AS Classificatie,
        w.Postcode,
        CASE
            WHEN m.Complexcode IS NULL AND c.Complexcode IS NULL THEN 'right_only'
            WHEN w."VHE-nr" IS NULL THEN 'left_only'
            ELSE 'both'
        END AS _merge
    FROM marktwaardeparameters m
    FULL OUTER JOIN complex_parameters c
        ON m.Complexcode = c.Complexcode
        AND m.Corporatie = c.Corporatie
        AND m.Jaar = c.Jaar
        AND m.Peildatum = c.Peildatum
    FULL OUTER JOIN {{ source('tms', 'valuation_overview_vhe') }} w
        ON m."VHE-nr" = w."VHE-nr"
        AND COALESCE(m.Corporatie, c.Corporatie) = w.Corporatie
        AND COALESCE(m.Jaar, c.Jaar) = w.Jaar
        AND COALESCE(m.Peildatum, c.Peildatum) = w.Peildatum
)
SELECT * FROM merge
