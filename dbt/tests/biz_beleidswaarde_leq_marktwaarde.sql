-- Business logic: Beleidswaarde must not exceed Marktwaarde.
-- Policy value is defined as market value minus policy discounts, so it should always be ≤ market value.
-- Small deviations (< €100) are acceptable due to rounding.
{{ config(severity='warn') }}
SELECT
    "VHE-nr",
    Corporatie,
    Jaar,
    Marktwaarde,
    Beleidswaarde,
    Beleidswaarde - Marktwaarde AS excess
FROM {{ ref('dataset_basis') }}
WHERE Beleidswaarde IS NOT NULL
  AND Marktwaarde IS NOT NULL
  AND Beleidswaarde > Marktwaarde + 100
