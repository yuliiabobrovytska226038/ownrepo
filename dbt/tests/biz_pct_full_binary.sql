-- Business logic: % Full must be strictly 0 or 1.
-- This flag is derived from freedom-degree overrides and must be binary.
SELECT
    "VHE-nr",
    Corporatie,
    Jaar,
    "% Full"
FROM {{ ref('stg_tms_percentage_full') }}
WHERE "% Full" IS NOT NULL
  AND "% Full" NOT IN (0, 1)
