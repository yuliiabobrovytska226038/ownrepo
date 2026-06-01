-- Business logic: Market value (Netto marktwaarde) must be non-negative.
-- Negative market values indicate data errors in the source system.
{{ config(severity='warn') }}
SELECT
    "VHE-nr",
    Corporatie,
    Jaar,
    Marktwaarde
FROM {{ ref('dataset_basis') }}
WHERE Marktwaarde IS NOT NULL
  AND Marktwaarde < 0
