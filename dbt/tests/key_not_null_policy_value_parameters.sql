-- Not-null check on join keys: VHE-nr, Corporatie, Peildatum in source policy_value_parameters.
-- These are the keys used to join into int_beleidswaarde and dataset_basis.
SELECT
    "VHE-nr",
    "Corporatie",
    "Peildatum"
FROM {{ source('tms', 'policy_value_parameters') }}
WHERE "VHE-nr" IS NULL
   OR "Corporatie" IS NULL
   OR "Peildatum" IS NULL
