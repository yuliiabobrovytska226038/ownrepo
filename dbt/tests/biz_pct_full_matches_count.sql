-- Business logic: Percentage Full consistency.
-- % Full should be 1 when Aantal vrijheidsgraden toegepast > 0, and 0 otherwise.
-- Detects inconsistencies between the freedom-degree count and the binary flag.
SELECT
    "VHE-nr",
    Corporatie,
    Jaar,
    "Aantal vrijheidsgraden toegepast",
    "% Full"
FROM {{ ref('dataset_basis') }}
WHERE "% Full" IS NOT NULL
  AND "Aantal vrijheidsgraden toegepast" IS NOT NULL
  AND (
    ("Aantal vrijheidsgraden toegepast" > 0 AND "% Full" != 1)
    OR ("Aantal vrijheidsgraden toegepast" = 0 AND "% Full" != 0)
  )
