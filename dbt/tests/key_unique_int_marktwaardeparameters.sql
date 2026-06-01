-- Composite key uniqueness on int_marktwaardeparameters output: VHE-nr + Corporatie + Jaar.
-- This model is joined into dataset_basis; duplicates would cause fanout.
SELECT
    "VHE-nr",
    "Corporatie",
    "Jaar",
    COUNT(*) AS row_count
FROM {{ ref('int_marktwaardeparameters') }}
WHERE "VHE-nr" IS NOT NULL
GROUP BY "VHE-nr", "Corporatie", "Jaar"
HAVING COUNT(*) > 1
