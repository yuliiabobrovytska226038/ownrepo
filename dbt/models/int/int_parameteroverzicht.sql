-- int_parameteroverzicht: VHE + Complex parameter overview joined
-- Recreates legacy vw_parameteroverzicht from duck_query.py

WITH merged AS (
    SELECT
        COALESCE(v.Corporatie, c.Corporatie) AS Corporatie,
        COALESCE(v.Jaar, c.Jaar) AS Jaar,
        COALESCE(v.Peildatum, c.Peildatum) AS Peildatum,
        COALESCE(v.Waarderingscomplex, c.Waarderingscomplex) AS Waarderingscomplex,
        v.* EXCLUDE (Corporatie, Jaar, Peildatum, Waarderingscomplex, Complexnaam, Model),
        c.* EXCLUDE (Corporatie, Jaar, Peildatum, Waarderingscomplex, Complexnaam, Straatnaam, Model),
        COALESCE(v.Complexnaam, c.Complexnaam) AS Complexnaam,
        CASE
            WHEN v.Waarderingscomplex IS NULL THEN 'right_only'
            WHEN c.Waarderingscomplex IS NULL THEN 'left_only'
            ELSE 'both'
        END AS _merge
    FROM {{ ref('int_parameteroverzicht_vhe') }} v
    FULL OUTER JOIN {{ ref('int_parameteroverzicht_complex') }} c
        ON v.Waarderingscomplex = c.Waarderingscomplex
        AND v.Corporatie = c.Corporatie
        AND v.Peildatum = c.Peildatum
)
SELECT *
FROM merged
