-- Not-null and uniqueness check on postcode in external.postal_code_mapping.
-- This is the join key used in dataset_basis for geographic enrichment.
SELECT
    postcode,
    COUNT(*) AS row_count
FROM {{ source('external', 'postal_code_mapping') }}
WHERE postcode IS NOT NULL
GROUP BY postcode
HAVING COUNT(*) > 1

UNION ALL

SELECT
    'NULL_POSTCODE' AS postcode,
    COUNT(*) AS row_count
FROM {{ source('external', 'postal_code_mapping') }}
WHERE postcode IS NULL
HAVING COUNT(*) > 0
