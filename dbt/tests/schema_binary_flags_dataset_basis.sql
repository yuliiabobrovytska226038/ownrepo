-- Schema: "% Full" and "% Vrije waardering" must be binary (0 or 1) when not null.
SELECT "VHE-nr", "Corporatie", "% Full", "% Vrije waardering"
FROM {{ ref('dataset_basis') }}
WHERE ("% Full" IS NOT NULL AND "% Full" NOT IN (0, 1))
   OR ("% Vrije waardering" IS NOT NULL AND "% Vrije waardering" NOT IN (0, 1))
