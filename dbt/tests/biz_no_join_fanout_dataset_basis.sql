-- Schema: Composite uniqueness on the primary key of dataset_basis.
-- No duplicate VHE-nr + Corporatie + Jaar combinations should exist (would indicate join fanout).
SELECT
    "VHE-nr",
    Corporatie,
    Jaar,
    COUNT(*) AS row_count
FROM {{ ref('dataset_basis') }}
GROUP BY "VHE-nr", Corporatie, Jaar
HAVING COUNT(*) > 1
