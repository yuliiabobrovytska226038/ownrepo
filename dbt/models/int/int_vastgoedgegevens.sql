-- int_vastgoedgegevens: VHE + Complex property data joined
-- Recreates legacy vw_vastgoedgegevens from duck_query.py

WITH merged AS (
    SELECT
        COALESCE(v.Corporatie, w.Corporatie) AS Corporatie,
        COALESCE(v.Jaar, w.Jaar) AS Jaar,
        COALESCE(v.Peildatum, w.Peildatum) AS Peildatum,
        v.* EXCLUDE (Corporatie, Jaar, Peildatum),
        w.* EXCLUDE (Corporatie, Jaar, Peildatum),
        CASE
            WHEN v.Waarderingscomplex IS NULL THEN 'right_only'
            WHEN w.Complexcode IS NULL THEN 'left_only'
            ELSE 'both'
        END AS _merge
    FROM {{ source('tms', 'vastgoedgegevens_vhe_gegevens') }} v
    FULL OUTER JOIN {{ source('tms', 'vastgoedgegevens_waarderingscomplexen') }} w
        ON v.Waarderingscomplex = w.Complexcode
        AND v.Corporatie = w.Corporatie
        AND v.Jaar = w.Jaar
        AND v.Peildatum = w.Peildatum
)
SELECT *
FROM merged
