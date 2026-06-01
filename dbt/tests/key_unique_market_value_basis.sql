-- Composite key uniqueness: rentalUnitInternalId + Corporatie + Jaar must be unique in source market_value_basis.
-- This key is used to join market_value_basis to rental_unit_mapping for VHE-nr resolution.
SELECT
    "rentalUnitInternalId",
    "Corporatie",
    "Jaar",
    COUNT(*) AS row_count
FROM {{ source('tms', 'market_value_basis') }}
GROUP BY "rentalUnitInternalId", "Corporatie", "Jaar"
HAVING COUNT(*) > 1
