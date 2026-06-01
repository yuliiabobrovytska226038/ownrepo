-- Schema: not_null checks on dataset_basis primary/join key columns.
SELECT *
FROM {{ ref('dataset_basis') }}
WHERE "VHE-nr" IS NULL
   OR "Waarderingscomplex" IS NULL
   OR "Corporatie" IS NULL
   OR "Waarderingsmodel" IS NULL
   OR "Handboektype" IS NULL
   OR "Jaar" IS NULL
   OR "Peildatum" IS NULL
