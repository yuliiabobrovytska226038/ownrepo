-- Composite key uniqueness: VHE-nummer + Corporatie + Jaar must be unique in source.
SELECT
    "VHE-nummer",
    "Corporatie",
    "Jaar",
    COUNT(*) AS row_count
FROM {{ source('tms', 'vastgoedgegevens_vhe_gegevens') }}
GROUP BY "VHE-nummer", "Corporatie", "Jaar"
HAVING COUNT(*) > 1
