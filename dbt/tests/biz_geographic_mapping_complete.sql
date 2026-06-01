-- Business logic: Geographic mapping completeness.
-- Verify that postcode → gemeente → COROP → province chain is complete for the majority of rows.
-- Rows with NULL geographic fields indicate broken postcode mapping.
{{ config(severity='warn') }}
SELECT
    "VHE-nr",
    Corporatie,
    Jaar,
    Postcode,
    Gemeentecode,
    "COROP-gebied",
    "Provincies Naam"
FROM {{ ref('dataset_basis') }}
WHERE Postcode IS NOT NULL
  AND (
    Gemeentecode IS NULL
    OR "COROP-gebied" IS NULL
    OR "Provincies Naam" IS NULL
  )
