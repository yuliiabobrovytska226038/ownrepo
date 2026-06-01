-- Composite key uniqueness on int_marktwaarde output: VHE-nr + Corporatie + Jaar.
-- Ensures no join fanout occurred during the three-way FULL OUTER JOIN.
SELECT
    "VHE-nr",
    "Corporatie",
    "Jaar",
    COUNT(*) AS row_count
FROM {{ ref('int_marktwaarde') }}
WHERE "VHE-nr" IS NOT NULL
GROUP BY "VHE-nr", "Corporatie", "Jaar"
HAVING COUNT(*) > 1
