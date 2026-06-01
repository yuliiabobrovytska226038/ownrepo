-- Business logic: Discount rate (Disconteringsvoet) must be within a reasonable range.
-- Expected range: 2% to 15% (0.02 to 0.15 as decimal, or 2.0 to 15.0 as percentage).
-- TMS stores values as percentages (e.g. 5.25 means 5.25%).
{{ config(severity='warn') }}
SELECT
    "VHE-nr",
    Corporatie,
    Jaar,
    Disconteringsvoet
FROM {{ ref('dataset_basis') }}
WHERE Disconteringsvoet IS NOT NULL
  AND (Disconteringsvoet < 2.0 OR Disconteringsvoet > 15.0)
