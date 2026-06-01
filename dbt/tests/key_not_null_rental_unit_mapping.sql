-- Not-null check on join keys: rentalUnitInternalId, Corporatie, Jaar in source rental_unit_mapping.
-- These are the keys used to join market_value_basis back to VHE-nr.
SELECT
    "rentalUnitInternalId",
    "VHE-nr",
    "Corporatie",
    "Jaar"
FROM {{ source('tms', 'rental_unit_mapping') }}
WHERE "rentalUnitInternalId" IS NULL
   OR "VHE-nr" IS NULL
   OR "Corporatie" IS NULL
   OR "Jaar" IS NULL
