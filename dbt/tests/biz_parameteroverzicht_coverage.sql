-- Business logic: Woningen VHEs in dataset_basis should have matching rows in int_parameteroverzicht.
-- Verifies join completeness between the main fact table and parameter overview for residential units.
{{ config(severity='warn') }}
SELECT
    b."VHE-nr",
    b.Corporatie,
    b.Jaar
FROM {{ ref('dataset_basis') }} b
LEFT JOIN {{ ref('int_parameteroverzicht') }} p
    ON b."VHE-nr" = p."VHE-nr"
    AND b.Corporatie = p.Corporatie
    AND b.Jaar = p.Jaar
WHERE b.Waarderingsmodel = 'Woningen'
  AND p."VHE-nr" IS NULL
