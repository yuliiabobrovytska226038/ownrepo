{#
    All 10 marktwaarde VHG corporatie boxplot datasets (niveau + aanpassing).
    Includes both Full and Full (handboek) for each VHG.
    Python filters by vhg_key + calc_type for each chart.
#}

{% set prior_year = var("jaar") - 1 %}

{% set vhgs = [
    {"key": "dv_de", "use_vhg": "Bron disconteringsvoet (DV)", "scenario": "Doorexploiteren", "value_vhg": "DV doorexploiteren", "value_basic": "DV doorexploiteren handboek", "value_diff": "Absoluut"},
    {"key": "dv_up", "use_vhg": "Bron disconteringsvoet (DV)", "scenario": "Uitponden", "value_vhg": "DV uitponden", "value_basic": "DV uitponden handboek", "value_diff": "Absoluut"},
    {"key": "lw", "use_vhg": "Bron leegwaarde (LW)", "scenario": "Uitponden", "value_vhg": "LW waarde", "value_basic": "LW handboek waarde", "value_diff": "Relatief"},
    {"key": "mh", "use_vhg": "Bron markthuur (MH)", "scenario": "Doorexploiteren", "value_vhg": "MH waarde", "value_basic": "MH handboek waarde", "value_diff": "Relatief"},
    {"key": "oh_de", "use_vhg": "Bron Onderhoud (OH)", "scenario": "Doorexploiteren", "value_vhg": "OH doorexploiteren", "value_basic": "OH doorexploiteren handboek", "value_diff": "Relatief"},
    {"key": "oh_up", "use_vhg": "Bron Onderhoud (OH)", "scenario": "Uitponden", "value_vhg": "OH uitponden", "value_basic": "OH uitponden handboek", "value_diff": "Relatief"},
    {"key": "mu_de", "use_vhg": "Bron mutatiegraad doorexploiteren (MD)", "scenario": "Doorexploiteren", "value_vhg": "MD waarde", "value_basic": "MD handboek waarde", "value_diff": "Absoluut"},
    {"key": "mu_up", "use_vhg": "Bron mutatiegraad uitponden (MU)", "scenario": "Uitponden", "value_vhg": "MU jaar 1-15", "value_basic": "MU jaar 1-15 handboek", "value_diff": "Absoluut"},
    {"key": "ey_de", "use_vhg": "Bron exit yield (EY)", "scenario": "Doorexploiteren", "value_vhg": "EY doorexploiteren", "value_basic": "EY doorexploiteren handboek", "value_diff": "Absoluut"},
    {"key": "ey_up", "use_vhg": "Bron exit yield (EY)", "scenario": "Uitponden", "value_vhg": "EY uitponden", "value_basic": "EY uitponden handboek", "value_diff": "Absoluut"},
] %}

{# Full (eigen invoer) rows #}
{% for vhg in vhgs %}
{% if not loop.first %}union all{% endif %}
{{ vhg_corporatie_value_pair(
    source_ref="chart_vhg__mw_prepared",
    vhg_key=vhg.key,
    use_vhg=vhg.use_vhg,
    scenario=vhg.scenario,
    value_vhg=vhg.value_vhg,
    value_diff=vhg.value_diff,
    b_full=true
) }}
{% endfor %}

{# Full (handboek) rows — use handboek column and relabel Waarderingstype #}
{% for vhg in vhgs %}
union all

    select
        '{{ vhg.key }}' as vhg_key,
        aggregated.corporatie as "Corporatie",
        'Full (handboek)' as "Waarderingstype",
        niveau_value as value,
        'niveau' as calc_type
    from (
        select
            corporatie,
            avg("{{ vhg.value_basic }}") as niveau_value
        from {{ ref("chart_vhg__mw_prepared") }}
        where 1=1
        {{ vhg_filter_where(use_vhg=vhg.use_vhg, scenario=vhg.scenario, b_full=true, value_vhg=vhg.value_basic) }}
        group by corporatie
    ) as aggregated

{% endfor %}
