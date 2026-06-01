-- Business logic: dataset_validatie should contain only Woningen rows.
-- Verifies via join back to dataset_basis (since Waarderingsmodel is excluded from the view).
SELECT
    v."VHE-nr",
    v.Corporatie,
    v.Jaar,
    b.Waarderingsmodel
FROM {{ ref('dataset_validatie') }} v
LEFT JOIN {{ ref('dataset_basis') }} b
    ON v."VHE-nr" = b."VHE-nr"
    AND v.Corporatie = b.Corporatie
    AND v.Jaar = b.Jaar
WHERE b.Waarderingsmodel != 'Woningen'
   OR b.Waarderingsmodel IS NULL
