-- Composite key uniqueness on int_beleidswaarde output: VHE-nr + Corporatie + Jaar.
-- Ensures no join fanout occurred during the FULL OUTER JOIN of policy_value_parameters and policy_value_report_vhe.
SELECT
    "VHE-nr",
    "Corporatie",
    "Jaar",
    COUNT(*) AS row_count
FROM {{ ref('int_beleidswaarde') }}
WHERE "VHE-nr" IS NOT NULL
GROUP BY "VHE-nr", "Corporatie", "Jaar"
HAVING COUNT(*) > 1
