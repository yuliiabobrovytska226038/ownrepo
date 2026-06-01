-- Business logic: Beleidswaarde discount rate must be within a reasonable range.
-- Expected range: 2% to 10% (stored as percentage, e.g. 5.25 means 5.25%).
{{ config(severity='warn') }}
SELECT
    "VHE-nr",
    Corporatie,
    Jaar,
    "Disconteringsvoet Beleidswaarde"
FROM {{ ref('dataset_basis') }}
WHERE "Disconteringsvoet Beleidswaarde" IS NOT NULL
  AND ("Disconteringsvoet Beleidswaarde" < 2.0 OR "Disconteringsvoet Beleidswaarde" > 10.0)
