-- Referential integrity: every VHE in int_marktwaarde must exist in int_vastgoedgegevens.
-- Detects broken joins between market value parameters and property data.
{{ config(severity='warn') }}
SELECT
    m."VHE-nr",
    m.Corporatie,
    m.Jaar
FROM {{ ref('int_marktwaarde') }} m
LEFT JOIN {{ ref('int_vastgoedgegevens') }} v
    ON m."VHE-nr" = v."VHE-nummer"
    AND m.Corporatie = v.Corporatie
    AND m.Jaar = v.Jaar
WHERE v."VHE-nummer" IS NULL
