-- dataset_validatie_aantallen: Statistics per filtering step
-- Recreates legacy dataset_validatie_aantallen from duck_query.py

{% set validation_steps = [
    ("Startstand", "base_data"),
    ("Corporatie percentage full >90%", "step_1_full_perc"),
    ("Geen (gemengde) complexen met studenten en extramurale zorgeenheden", "step_2_gemengd_complex"),
    ("Geen aardbevings- en/of krimpgebieden", "step_3_aardbeving_krimp"),
    ("Geen jaarlijkse erfpacht", "step_4_erfpacht"),
    ("Minimaal aantal eenheden 250", "step_5_min_250_vhes"),
] %}

WITH {{ dataset_validatie_filter_ctes() }},
aantallen AS (
    {% for label, step_relation in validation_steps %}
    SELECT
        '{{ label }}' AS "Omschrijving",
        COUNT(DISTINCT(Corporatie)) AS Corporaties,
        COUNT(DISTINCT(Corporatie || Werkmaatschappij)) AS Werkmaatschappijen,
        COUNT(DISTINCT(Corporatie || Werkmaatschappij || Waarderingscomplex)) AS Complexen,
        COUNT(*) AS VHEs,
        ROUND(SUM(Marktwaarde) / 1000000000, 1) AS "Marktwaarde (mlrd)"
    FROM {{ step_relation }}
    {% if not loop.last %}UNION ALL{% endif %}
    {% endfor %}
)
SELECT * FROM aantallen
