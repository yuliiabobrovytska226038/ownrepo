-- Composite key uniqueness: VHE-nr + Corporatie + Peildatum must be unique in source policy_value_parameters.
-- This key is used in the FULL OUTER JOIN in int_beleidswaarde and LEFT JOIN in dataset_basis.
SELECT
    "VHE-nr",
    "Corporatie",
    "Peildatum",
    COUNT(*) AS row_count
FROM {{ source('tms', 'policy_value_parameters') }}
GROUP BY "VHE-nr", "Corporatie", "Peildatum"
HAVING COUNT(*) > 1
