-- Business logic: Free-valuation counts must be internally consistent.
-- VHEs with Waarderingstype = 'Vrije waardering' must have % Vrije waardering = 1,
-- and % Vrije waardering = 1 must imply Waarderingstype = 'Vrije waardering'.
SELECT
    "VHE-nr",
    Corporatie,
    Jaar,
    Waarderingstype,
    "% Vrije waardering"
FROM {{ ref('stg_tms_percentage_full') }}
WHERE (Waarderingstype = 'Vrije waardering' AND "% Vrije waardering" != 1)
   OR ("% Vrije waardering" = 1 AND Waarderingstype != 'Vrije waardering')
