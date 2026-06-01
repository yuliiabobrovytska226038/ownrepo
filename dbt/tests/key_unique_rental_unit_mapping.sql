-- Composite key uniqueness: rentalUnitInternalId + Corporatie + Jaar must be unique in source.
SELECT
    "rentalUnitInternalId",
    "Corporatie",
    "Jaar",
    COUNT(*) AS row_count
FROM {{ source('tms', 'rental_unit_mapping') }}
GROUP BY "rentalUnitInternalId", "Corporatie", "Jaar"
HAVING COUNT(*) > 1
