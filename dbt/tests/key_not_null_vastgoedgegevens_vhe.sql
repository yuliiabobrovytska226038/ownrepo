-- Not-null check on join keys: VHE-nummer, Corporatie, Jaar in source vastgoedgegevens_vhe_gegevens.
-- These are the keys used to join into int_vastgoedgegevens and dataset_basis.
SELECT
    "VHE-nummer",
    "Corporatie",
    "Jaar"
FROM {{ source('tms', 'vastgoedgegevens_vhe_gegevens') }}
WHERE "VHE-nummer" IS NULL
   OR "Corporatie" IS NULL
   OR "Jaar" IS NULL
