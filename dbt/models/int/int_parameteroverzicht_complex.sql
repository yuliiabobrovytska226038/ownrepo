-- int_parameteroverzicht_complex: Complex-level parameter overview
-- Recreates legacy Parameteroverzicht_Complex intermediate table from duck_query.py

{% set complex_parameter_exclusions = 'Corporatie, Jaar, Peildatum, Waarderingscomplex, Complexnaam, Straatnaam, Model' %}

WITH merged AS (
    SELECT
        ms.*,
        ls.* EXCLUDE ({{ complex_parameter_exclusions }}),
        mg.* EXCLUDE ({{ complex_parameter_exclusions }}),
        oe.* EXCLUDE ({{ complex_parameter_exclusions }}),
        oko.* EXCLUDE ({{ complex_parameter_exclusions }}),
        sc.* EXCLUDE ({{ complex_parameter_exclusions }}),
        sk.* EXCLUDE ({{ complex_parameter_exclusions }}),
        vb.* EXCLUDE ({{ complex_parameter_exclusions }})
    FROM {{ source('tms', 'parameters_overview_markthuurstijging') }} ms
    FULL OUTER JOIN {{ source('tms', 'parameters_overview_leegwaardestijging') }} ls
        ON ms.Waarderingscomplex = ls.Waarderingscomplex AND ms.Corporatie = ls.Corporatie AND ms.Peildatum = ls.Peildatum
    FULL OUTER JOIN {{ source('tms', 'parameters_overview_mutatiegraad') }} mg
        ON ms.Waarderingscomplex = mg.Waarderingscomplex AND ms.Corporatie = mg.Corporatie AND ms.Peildatum = mg.Peildatum
    FULL OUTER JOIN {{ source('tms', 'parameters_overview_overige_exploitatielasten') }} oe
        ON ms.Waarderingscomplex = oe.Waarderingscomplex AND ms.Corporatie = oe.Corporatie AND ms.Peildatum = oe.Peildatum
    FULL OUTER JOIN {{ source('tms', 'parameters_overview_overige_kosten_en_opbrengsten') }} oko
        ON ms.Waarderingscomplex = oko.Waarderingscomplex AND ms.Corporatie = oko.Corporatie AND ms.Peildatum = oko.Peildatum
    FULL OUTER JOIN {{ source('tms', 'parameters_overview_scenario') }} sc
        ON ms.Waarderingscomplex = sc.Waarderingscomplex AND ms.Corporatie = sc.Corporatie AND ms.Peildatum = sc.Peildatum
    FULL OUTER JOIN {{ source('tms', 'parameters_overview_splitsingskosten') }} sk
        ON ms.Waarderingscomplex = sk.Waarderingscomplex AND ms.Corporatie = sk.Corporatie AND ms.Peildatum = sk.Peildatum
    FULL OUTER JOIN {{ source('tms', 'parameters_overview_verkoopbeperking_woningen') }} vb
        ON ms.Waarderingscomplex = vb.Waarderingscomplex AND ms.Corporatie = vb.Corporatie AND ms.Peildatum = vb.Peildatum
)
SELECT * FROM merged
