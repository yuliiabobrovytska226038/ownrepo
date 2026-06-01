-- Referential integrity: VHEs in int_parameteroverzicht should exist in int_vastgoedgegevens.
-- Detects broken joins between parameter overview and property data.
{{ config(severity='warn') }}
SELECT
    p."VHE-nr",
    p.Corporatie,
    p.Peildatum
FROM {{ ref('int_parameteroverzicht') }} p
LEFT JOIN {{ ref('int_vastgoedgegevens') }} v
    ON p."VHE-nr" = v."VHE-nummer"
    AND p.Corporatie = v.Corporatie
    AND p.Peildatum = v.Peildatum
WHERE p."VHE-nr" IS NOT NULL
  AND v."VHE-nummer" IS NULL
