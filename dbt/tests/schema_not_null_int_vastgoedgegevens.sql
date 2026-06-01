-- Schema: not_null checks on int_vastgoedgegevens join keys.
-- VHE-nummer may be null for complex-only records (FULL OUTER JOIN), so only check Corporatie/Jaar/Peildatum.
SELECT *
FROM {{ ref('int_vastgoedgegevens') }}
WHERE "Corporatie" IS NULL
   OR "Jaar" IS NULL
   OR "Peildatum" IS NULL
