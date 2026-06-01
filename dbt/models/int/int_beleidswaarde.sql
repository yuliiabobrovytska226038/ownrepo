-- int_beleidswaarde: Policy value params + report joined
-- Recreates legacy vw_beleidswaarde from duck_query.py

WITH merged AS (
    SELECT
        COALESCE(p."VHE-nr", v."VHE-nr") AS "VHE-nr",
        COALESCE(p.Corporatie, v.Corporatie) AS Corporatie,
        COALESCE(p.Jaar, v.Jaar) AS Jaar,
        COALESCE(p.Peildatum, v.Peildatum) AS Peildatum,
        p.* EXCLUDE ("VHE-nr", Corporatie, Jaar, Peildatum, Complexnaam, Deelportefeuille, Classificatie, Adres, Beleidshuur, Beleidsbeheer),
        v.* EXCLUDE ("VHE-nr", Corporatie, Jaar, Peildatum),
        CASE
            WHEN p."VHE-nr" IS NULL THEN 'right_only'
            WHEN v."VHE-nr" IS NULL THEN 'left_only'
            ELSE 'both'
        END AS _merge
    FROM {{ source('tms', 'policy_value_parameters') }} p
    FULL OUTER JOIN {{ source('tms', 'policy_value_report_vhe') }} v
        ON p."VHE-nr" = v."VHE-nr"
        AND p.Corporatie = v.Corporatie
        AND p.Jaar = v.Jaar
        AND p.Peildatum = v.Peildatum
)
SELECT *
FROM merged
