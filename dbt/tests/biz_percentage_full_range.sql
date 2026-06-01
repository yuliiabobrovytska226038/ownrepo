-- Business logic: Percentage-based columns in dataset_basis must be within [0, 1] range.
-- Checks Beschikbaarheid and Kwaliteit scores don't exceed market value.
{{ config(severity='warn') }}
SELECT
    "VHE-nr",
    Corporatie,
    Jaar,
    Beschikbaarheid_totaal,
    Kwaliteit_totaal,
    Betaalbaarheid,
    Beheer,
    Marktwaarde
FROM {{ ref('dataset_basis') }}
WHERE Marktwaarde IS NOT NULL
  AND Marktwaarde > 0
  AND (
    (Beschikbaarheid_totaal IS NOT NULL AND Beschikbaarheid_totaal < 0)
    OR (Kwaliteit_totaal IS NOT NULL AND Kwaliteit_totaal < 0)
    OR (Betaalbaarheid IS NOT NULL AND Betaalbaarheid < 0)
    OR (Beheer IS NOT NULL AND Beheer < 0)
  )
