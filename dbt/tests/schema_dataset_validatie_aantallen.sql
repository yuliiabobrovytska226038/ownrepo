-- Schema: not_null and unique on Omschrijving + not_null on all columns in dataset_validatie_aantallen.
SELECT 'null_omschrijving' AS issue, "Omschrijving"
FROM {{ ref('dataset_validatie_aantallen') }}
WHERE "Omschrijving" IS NULL
   OR "Corporaties" IS NULL
   OR "Werkmaatschappijen" IS NULL
   OR "Complexen" IS NULL
   OR "VHEs" IS NULL
   OR "Marktwaarde (mlrd)" IS NULL

UNION ALL

SELECT 'duplicate_omschrijving' AS issue, "Omschrijving"
FROM {{ ref('dataset_validatie_aantallen') }}
GROUP BY "Omschrijving"
HAVING COUNT(*) > 1
