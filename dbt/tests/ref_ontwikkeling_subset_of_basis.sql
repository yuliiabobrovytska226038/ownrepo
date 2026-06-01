-- Referential integrity: every current-year VHE in dataset_ontwikkeling must exist in dataset_basis.
-- Excludes right_only rows (prior-year-only VHEs that left the portfolio).
SELECT
    o."VHE-nr",
    o.Corporatie,
    o.Jaar
FROM {{ ref('dataset_ontwikkeling') }} o
LEFT JOIN {{ ref('dataset_basis') }} b
    ON o."VHE-nr" = b."VHE-nr"
    AND o.Corporatie = b.Corporatie
    AND o.Jaar = b.Jaar
WHERE b."VHE-nr" IS NULL
  AND o."_merge" != 'right_only'
