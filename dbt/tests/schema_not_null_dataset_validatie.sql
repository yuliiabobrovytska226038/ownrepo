-- Schema: not_null checks on dataset_validatie key columns.
SELECT *
FROM {{ ref('dataset_validatie') }}
WHERE "VHE-nr" IS NULL
   OR "Waarderingscomplex" IS NULL
   OR "Corporatie" IS NULL
   OR "Handboektype" IS NULL
   OR "Jaar" IS NULL
   OR "Peildatum" IS NULL
