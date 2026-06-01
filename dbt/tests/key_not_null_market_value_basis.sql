-- Not-null check on join keys: rentalUnitInternalId, Corporatie, Jaar in source market_value_basis.
-- These are the keys used to join to rental_unit_mapping for VHE-nr resolution.
SELECT
    "rentalUnitInternalId",
    "Corporatie",
    "Jaar"
FROM {{ source('tms', 'market_value_basis') }}
WHERE "rentalUnitInternalId" IS NULL
   OR "Corporatie" IS NULL
   OR "Jaar" IS NULL
