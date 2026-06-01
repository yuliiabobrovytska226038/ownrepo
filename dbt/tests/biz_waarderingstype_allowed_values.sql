-- Schema: Waarderingstype must be one of the three allowed valuation types.
-- Only Basis, Full, or Vrije waardering are valid classification values.
SELECT
    "VHE-nr",
    Corporatie,
    Jaar,
    Waarderingstype
FROM {{ ref('stg_tms_percentage_full') }}
WHERE Waarderingstype IS NOT NULL
  AND Waarderingstype NOT IN ('Basis', 'Full', 'Vrije waardering')
