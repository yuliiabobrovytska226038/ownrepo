-- Not-null check on join keys: VHE-nr, Corporatie, Jaar in source valuation_overview_vhe.
-- These are the keys used to join into int_marktwaarde and dataset_basis.
SELECT
    "VHE-nr",
    "Corporatie",
    "Jaar"
FROM {{ source('tms', 'valuation_overview_vhe') }}
WHERE "VHE-nr" IS NULL
   OR "Corporatie" IS NULL
   OR "Jaar" IS NULL
