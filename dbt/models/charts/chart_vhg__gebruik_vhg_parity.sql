{% set vrijheidsgraad_sources = [
    ("Disconteringsvoet", "Bron disconteringsvoet (DV)"),
    ("Erfpacht", "Erfpacht (EP)"),
    ("Exit yield", "Bron exit yield (EY)"),
    ("Leegwaarde", "Bron leegwaarde (LW)"),
    ("Markthuur", "Bron markthuur (MH)"),
    ("Onderhoud", "Bron Onderhoud (OH)"),
    ("Mutatieonderhoud", "Bron mutatieonderhoud (MOH)"),
    ("Markthuurstijging", "Bron markthuurstijging (MHS)"),
    ("Leegwaardestijging verleden", "Bron leegwaardestijging verleden (LSV)"),
    ("Leegwaardestijging", "Bron leegwaardestijging (LS)"),
    ("Mutatiegraad doorexploiteren", "Bron mutatiegraad doorexploiteren (MD)"),
    ("Mutatiegraad uitponden", "Bron mutatiegraad uitponden (MU)"),
    ("Overige kosten", "Bron overige kosten (OVK)"),
    ("Overige opbrengsten", "Bron overige opbrengsten (OVO)"),
    ("Scenario", "Bron scenario (SC)"),
    ("Technische splitsingskosten", "Bron technische splitsingskosten (TS)"),
    ("Gedeelte niet verkopen bij mutatie", "Bron gedeelte niet verkopen bij mutatie (VM)"),
] %}

with source as (
    select *
    from {{ ref("chart_vhg__mw_prepared") }}
    where "Waarderingstype" = 'Full'
),

unpivoted as (
    select
        vrijheidsgraad as "Vrijheidsgraad",
        handboektype,
        case when bron in ('Eigen invoer', 'VTW') then 1.0 else 0.0 end as toegepast
    from source
    cross join lateral (
        values
            {% for label, column in vrijheidsgraad_sources %}
            {% if not loop.first %},{% endif %}
            ('{{ label }}', "{{ column }}")
            {% endfor %}
    ) as vrijheidsgraad_values(vrijheidsgraad, bron)
),

grouped as (
    select
        "Vrijheidsgraad",
        avg(case when handboektype = 'EGW' then toegepast end) as "EGW",
        avg(case when handboektype = 'MGW' then toegepast end) as "MGW",
        avg(toegepast) as "Totaal"
    from unpivoted
    group by "Vrijheidsgraad"
)

select
    "Vrijheidsgraad",
    "EGW",
    "MGW",
    "Totaal",
    {{ format_percent_text('"Totaal"') }} as "Text"
from grouped
order by "Totaal" desc
