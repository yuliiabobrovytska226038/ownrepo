-- Business logic: Year-over-year beleidswaarde change should be within reasonable bounds.
-- Extreme YoY changes (> 100% increase or > 50% decrease) likely indicate data errors.
{{ config(severity='warn') }}
SELECT
    o."VHE-nr",
    o.Corporatie,
    o.Jaar,
    o.Beleidswaarde AS beleidswaarde_current,
    o."Beleidswaarde_{{ var('jaar') - 1 }}" AS beleidswaarde_previous,
    CASE
        WHEN o."Beleidswaarde_{{ var('jaar') - 1 }}" > 0
        THEN (o.Beleidswaarde - o."Beleidswaarde_{{ var('jaar') - 1 }}") / o."Beleidswaarde_{{ var('jaar') - 1 }}"
        ELSE NULL
    END AS pct_change
FROM {{ ref('dataset_ontwikkeling') }} o
WHERE o.Beleidswaarde IS NOT NULL
  AND o."Beleidswaarde_{{ var('jaar') - 1 }}" IS NOT NULL
  AND o."Beleidswaarde_{{ var('jaar') - 1 }}" > 0
  AND (
    (o.Beleidswaarde - o."Beleidswaarde_{{ var('jaar') - 1 }}") / o."Beleidswaarde_{{ var('jaar') - 1 }}" > 1.0
    OR (o.Beleidswaarde - o."Beleidswaarde_{{ var('jaar') - 1 }}") / o."Beleidswaarde_{{ var('jaar') - 1 }}" < -0.5
  )
