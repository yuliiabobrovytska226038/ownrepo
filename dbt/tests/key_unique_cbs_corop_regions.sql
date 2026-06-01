-- Not-null and uniqueness check on gemeentecode in external.cbs_corop_regions.
-- This is the join key used in dataset_basis for COROP region mapping.
SELECT
    gemeentecode,
    COUNT(*) AS row_count
FROM {{ source('external', 'cbs_corop_regions') }}
WHERE gemeentecode IS NOT NULL
GROUP BY gemeentecode
HAVING COUNT(*) > 1

UNION ALL

SELECT
    'NULL_GEMEENTECODE' AS gemeentecode,
    COUNT(*) AS row_count
FROM {{ source('external', 'cbs_corop_regions') }}
WHERE gemeentecode IS NULL
HAVING COUNT(*) > 0
