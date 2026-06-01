-- Composite key uniqueness: VHE-nr + Corporatie + Jaar must be unique in source.
SELECT
    "VHE-nr",
    "Corporatie",
    "Jaar",
    COUNT(*) AS row_count
FROM {{ source('tms', 'valuation_overview_vhe') }}
GROUP BY "VHE-nr", "Corporatie", "Jaar"
HAVING COUNT(*) > 1
