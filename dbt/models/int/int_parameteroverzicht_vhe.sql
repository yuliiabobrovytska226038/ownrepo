-- int_parameteroverzicht_vhe: VHE-level parameter overview
-- Recreates legacy Parameteroverzicht_VHE intermediate table from duck_query.py

{% set vhe_parameter_exclusions = 'Corporatie, Jaar, Peildatum, "VHE-nr", Adres, Model, Handboektype, Classificatie, Waarderingscomplex, Complexnaam' %}

WITH erfpacht AS (
    SELECT * FROM {{ source('tms', 'parameters_overview_erfpacht_bog_mog_zog') }}
    UNION ALL BY NAME
    SELECT * FROM {{ source('tms', 'parameters_overview_erfpacht_woningen_parkeren') }}
),
markthuur AS (
    SELECT * FROM {{ source('tms', 'parameters_overview_markthuur_bog_mog_zog') }}
    UNION ALL BY NAME
    SELECT * FROM {{ source('tms', 'parameters_overview_markthuur_woningen_parkeren') }}
),
onderhoud AS (
    SELECT * FROM {{ source('tms', 'parameters_overview_onderhoud_bog_mog_zog') }}
    UNION ALL BY NAME
    SELECT * FROM {{ source('tms', 'parameters_overview_onderhoud_woningen_parkeren') }}
),
merged AS (
    SELECT
        d.*,
        e.* EXCLUDE ({{ vhe_parameter_exclusions }}),
        ep.* EXCLUDE ({{ vhe_parameter_exclusions }}),
        ey.* EXCLUDE ({{ vhe_parameter_exclusions }}),
        h.* EXCLUDE ({{ vhe_parameter_exclusions }}),
        l.* EXCLUDE ({{ vhe_parameter_exclusions }}),
        lw.* EXCLUDE ({{ vhe_parameter_exclusions }}),
        m.* EXCLUDE ({{ vhe_parameter_exclusions }}),
        mh.* EXCLUDE ({{ vhe_parameter_exclusions }}),
        o.* EXCLUDE ({{ vhe_parameter_exclusions }}),
        s.* EXCLUDE ({{ vhe_parameter_exclusions }})
    FROM {{ source('tms', 'parameters_overview_disconteringsvoet') }} d
    FULL OUTER JOIN {{ source('tms', 'parameters_overview_epv') }} e
        ON d."VHE-nr" = e."VHE-nr" AND d.Corporatie = e.Corporatie AND d.Peildatum = e.Peildatum
    FULL OUTER JOIN erfpacht ep
        ON d."VHE-nr" = ep."VHE-nr" AND d.Corporatie = ep.Corporatie AND d.Peildatum = ep.Peildatum
    FULL OUTER JOIN {{ source('tms', 'parameters_overview_exit_yield') }} ey
        ON d."VHE-nr" = ey."VHE-nr" AND d.Corporatie = ey.Corporatie AND d.Peildatum = ey.Peildatum
    FULL OUTER JOIN {{ source('tms', 'parameters_overview_huurbeklemming') }} h
        ON d."VHE-nr" = h."VHE-nr" AND d.Corporatie = h.Corporatie AND d.Peildatum = h.Peildatum
    FULL OUTER JOIN {{ source('tms', 'parameters_overview_leegstand_woningen_parkeren') }} l
        ON d."VHE-nr" = l."VHE-nr" AND d.Corporatie = l.Corporatie AND d.Peildatum = l.Peildatum
    FULL OUTER JOIN {{ source('tms', 'parameters_overview_leegwaarde') }} lw
        ON d."VHE-nr" = lw."VHE-nr" AND d.Corporatie = lw.Corporatie AND d.Peildatum = lw.Peildatum
    FULL OUTER JOIN markthuur m
        ON d."VHE-nr" = m."VHE-nr" AND d.Corporatie = m.Corporatie AND d.Peildatum = m.Peildatum
    FULL OUTER JOIN {{ source('tms', 'parameters_overview_maximale_huur') }} mh
        ON d."VHE-nr" = mh."VHE-nr" AND d.Corporatie = mh.Corporatie AND d.Peildatum = mh.Peildatum
    FULL OUTER JOIN onderhoud o
        ON d."VHE-nr" = o."VHE-nr" AND d.Corporatie = o.Corporatie AND d.Peildatum = o.Peildatum
    FULL OUTER JOIN {{ source('tms', 'parameters_overview_schem_vrijheidsgraden_bog') }} s
        ON d."VHE-nr" = s."VHE-nr" AND d.Corporatie = s.Corporatie AND d.Peildatum = s.Peildatum
)
SELECT * FROM merged
