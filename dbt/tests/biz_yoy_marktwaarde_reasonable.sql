-- Business logic: Year-over-year market value change should be within reasonable bounds.
-- Extreme YoY changes (> 100% increase or > 50% decrease) likely indicate data errors.
{{ config(severity='warn') }}
SELECT
    o."VHE-nr",
    o.Corporatie,
    o.Jaar,
    o.Marktwaarde AS marktwaarde_current,
    o."Marktwaarde_{{ var('jaar') - 1 }}" AS marktwaarde_previous,
    CASE
        WHEN o."Marktwaarde_{{ var('jaar') - 1 }}" > 0
        THEN (o.Marktwaarde - o."Marktwaarde_{{ var('jaar') - 1 }}") / o."Marktwaarde_{{ var('jaar') - 1 }}"
        ELSE NULL
    END AS pct_change
FROM {{ ref('dataset_ontwikkeling') }} o
WHERE o.Marktwaarde IS NOT NULL
  AND o."Marktwaarde_{{ var('jaar') - 1 }}" IS NOT NULL
  AND o."Marktwaarde_{{ var('jaar') - 1 }}" > 0
  AND (
    (o.Marktwaarde - o."Marktwaarde_{{ var('jaar') - 1 }}") / o."Marktwaarde_{{ var('jaar') - 1 }}" > 1.0
    OR (o.Marktwaarde - o."Marktwaarde_{{ var('jaar') - 1 }}") / o."Marktwaarde_{{ var('jaar') - 1 }}" < -0.5
  )
