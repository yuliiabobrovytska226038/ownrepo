-- Schema: not_null checks on dataset_ontwikkeling key columns (current-year rows only).
-- Excludes right_only rows where current-year data is structurally null.
SELECT *
FROM {{ ref('dataset_ontwikkeling') }}
WHERE "_merge" != 'right_only'
  AND (
    "VHE-nr" IS NULL
    OR "Waarderingscomplex" IS NULL
    OR "Corporatie" IS NULL
    OR "Waarderingsmodel" IS NULL
    OR "Handboektype" IS NULL
    OR "Jaar" IS NULL
    OR "Peildatum" IS NULL
  )
