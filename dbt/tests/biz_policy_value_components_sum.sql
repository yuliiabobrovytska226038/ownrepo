-- Business logic: Policy value components must sum to total discount.
-- Beleidswaarde = Marktwaarde - Beschikbaarheid_totaal - Betaalbaarheid - Kwaliteit_totaal - Beheer
-- Tolerance: ±€10 for rounding differences.
{{ config(severity='warn') }}
SELECT
    "VHE-nr",
    Corporatie,
    Jaar,
    Marktwaarde,
    Beleidswaarde,
    Beschikbaarheid_totaal,
    Betaalbaarheid,
    Kwaliteit_totaal,
    Beheer,
    Marktwaarde - Beschikbaarheid_totaal - Betaalbaarheid - Kwaliteit_totaal - Beheer AS expected_bw,
    ABS((Marktwaarde - Beschikbaarheid_totaal - Betaalbaarheid - Kwaliteit_totaal - Beheer) - Beleidswaarde) AS diff
FROM {{ ref('dataset_basis') }}
WHERE Marktwaarde IS NOT NULL
  AND Beleidswaarde IS NOT NULL
  AND Beschikbaarheid_totaal IS NOT NULL
  AND Betaalbaarheid IS NOT NULL
  AND Kwaliteit_totaal IS NOT NULL
  AND Beheer IS NOT NULL
  AND ABS((Marktwaarde - Beschikbaarheid_totaal - Betaalbaarheid - Kwaliteit_totaal - Beheer) - Beleidswaarde) > 10
