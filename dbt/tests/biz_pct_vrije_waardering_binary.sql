-- Business logic: % Vrije waardering must be strictly 0 or 1.
-- This flag is derived from free-valuation override indicators and must be binary.
SELECT
    "VHE-nr",
    Corporatie,
    Jaar,
    "% Vrije waardering"
FROM {{ ref('stg_tms_percentage_full') }}
WHERE "% Vrije waardering" IS NOT NULL
  AND "% Vrije waardering" NOT IN (0, 1)
