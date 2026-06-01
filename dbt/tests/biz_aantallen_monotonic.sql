-- Business logic: Validation counts must be monotonically decreasing.
-- The filter funnel removes rows at each step, so VHE counts must decrease or stay equal.
-- Startstand >= step_1 >= step_2 >= step_3 >= step_4 >= step_5.
WITH ordered AS (
    SELECT
        "Omschrijving",
        "VHEs",
        ROW_NUMBER() OVER (ORDER BY
            CASE "Omschrijving"
                WHEN 'Startstand' THEN 1
                WHEN 'Corporatie percentage full >90%' THEN 2
                WHEN 'Geen (gemengde) complexen met studenten en extramurale zorgeenheden' THEN 3
                WHEN 'Geen aardbevings- en/of krimpgebieden' THEN 4
                WHEN 'Geen jaarlijkse erfpacht' THEN 5
                WHEN 'Minimaal aantal eenheden 250' THEN 6
            END
        ) AS step_order
    FROM {{ ref('dataset_validatie_aantallen') }}
),
violations AS (
    SELECT
        curr."Omschrijving" AS current_step,
        curr."VHEs" AS current_count,
        prev."Omschrijving" AS previous_step,
        prev."VHEs" AS previous_count
    FROM ordered curr
    JOIN ordered prev ON curr.step_order = prev.step_order + 1
    WHERE curr."VHEs" > prev."VHEs"
)
SELECT * FROM violations
