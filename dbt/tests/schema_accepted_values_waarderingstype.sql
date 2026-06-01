-- Schema: accepted_values check for Waarderingstype in dataset_basis.
SELECT "VHE-nr", "Corporatie", "Waarderingstype"
FROM {{ ref('dataset_basis') }}
WHERE "Waarderingstype" IS NOT NULL
  AND "Waarderingstype" NOT IN ('Basis', 'Full', 'Vrije waardering')
