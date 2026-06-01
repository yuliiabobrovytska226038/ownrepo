-- int_marktwaardeparameters: VHE params + Complex params merged
-- Recreates legacy vw_marktwaardeparameters from duck_query.py

WITH vhe_parameters AS (
    {{ market_value_vhe_parameter_sources() }}
),
complex_parameters AS (
    {{ market_value_complex_parameter_sources() }}
),
merged AS (
    SELECT
        COALESCE(v.Corporatie, c.Corporatie) AS Corporatie,
        COALESCE(v.Jaar, c.Jaar) AS Jaar,
        COALESCE(v.Peildatum, c.Peildatum) AS Peildatum,
        COALESCE(v.Complexcode, c.Complexcode) AS Complexcode,
        v.* EXCLUDE (Corporatie, Jaar, Peildatum, Complexcode),
        c.* EXCLUDE (Corporatie, Jaar, Peildatum, Complexcode, Complexnaam, Waarderingsmodel, Deelportefeuille),
        CASE
            WHEN v.Complexcode IS NULL THEN 'right_only'
            WHEN c.Complexcode IS NULL THEN 'left_only'
            ELSE 'both'
        END AS _merge
    FROM vhe_parameters v
    FULL OUTER JOIN complex_parameters c
        ON v.Complexcode = c.Complexcode
        AND v.Corporatie = c.Corporatie
        AND v.Jaar = c.Jaar
        AND v.Peildatum = c.Peildatum
)
SELECT *
FROM merged
