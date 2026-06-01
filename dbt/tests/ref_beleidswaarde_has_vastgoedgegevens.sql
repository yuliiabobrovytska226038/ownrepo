-- Referential integrity: every VHE in int_beleidswaarde must exist in int_vastgoedgegevens.
-- Detects broken joins between policy value data and property data.
{{ config(severity='warn') }}
SELECT
    b."VHE-nr",
    b.Corporatie,
    b.Jaar
FROM {{ ref('int_beleidswaarde') }} b
LEFT JOIN {{ ref('int_vastgoedgegevens') }} v
    ON b."VHE-nr" = v."VHE-nummer"
    AND b.Corporatie = v.Corporatie
    AND b.Jaar = v.Jaar
WHERE v."VHE-nummer" IS NULL
