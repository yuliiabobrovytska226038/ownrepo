-- Composite key uniqueness on stg_tms_percentage_full: VHE-nr + Corporatie + Jaar.
-- This staging model is joined into dataset_basis; duplicates would cause fanout.
SELECT
    "VHE-nr",
    "Corporatie",
    "Jaar",
    COUNT(*) AS row_count
FROM {{ ref('stg_tms_percentage_full') }}
WHERE "VHE-nr" IS NOT NULL
GROUP BY "VHE-nr", "Corporatie", "Jaar"
HAVING COUNT(*) > 1
