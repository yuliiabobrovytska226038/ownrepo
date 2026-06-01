-- Composite key uniqueness on int_parameteroverzicht output: VHE-nr + Corporatie + Peildatum.
-- Ensures no join fanout occurred during the FULL OUTER JOIN of VHE-level and complex-level parameter sheets.
SELECT
    "VHE-nr",
    "Corporatie",
    "Peildatum",
    COUNT(*) AS row_count
FROM {{ ref('int_parameteroverzicht') }}
WHERE "VHE-nr" IS NOT NULL
GROUP BY "VHE-nr", "Corporatie", "Peildatum"
HAVING COUNT(*) > 1
