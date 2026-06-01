-- Business logic: dataset_validatie must be a proper subset of dataset_basis.
-- Every row in the validation set should exist in the base dataset.
SELECT
    v."VHE-nr",
    v.Corporatie,
    v.Jaar
FROM {{ ref('dataset_validatie') }} v
LEFT JOIN {{ ref('dataset_basis') }} b
    ON v."VHE-nr" = b."VHE-nr"
    AND v.Corporatie = b.Corporatie
    AND v.Jaar = b.Jaar
WHERE b."VHE-nr" IS NULL
