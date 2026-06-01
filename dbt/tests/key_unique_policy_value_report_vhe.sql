-- Composite key uniqueness: VHE-nr + Corporatie + Peildatum must be unique in source.
SELECT
    "VHE-nr",
    "Corporatie",
    "Peildatum",
    COUNT(*) AS row_count
FROM {{ source('tms', 'policy_value_report_vhe') }}
GROUP BY "VHE-nr", "Corporatie", "Peildatum"
HAVING COUNT(*) > 1
