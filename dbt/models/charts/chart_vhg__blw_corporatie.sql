{#
    All 4 beleidswaarde VHG corporatie boxplot datasets (niveau + aanpassing).
    Python filters by vhg_key + calc_type for each chart.
#}

{% set vhgs = [
    {"key": "sh", "value_vhg": "Beleidshuur", "value_diff": "Relatief"},
    {"key": "bb", "value_vhg": "Beleidsbeheer", "value_diff": "Relatief"},
    {"key": "bo", "value_vhg": "Beleidsonderhoud", "value_diff": "Relatief"},
    {"key": "dv", "value_vhg": "DV doorexploiteren", "value_diff": "Relatief"},
] %}

{% for vhg in vhgs %}
{% if not loop.first %}union all{% endif %}
{{ vhg_corporatie_value_pair(
    source_ref="chart_vhg__blw_prepared",
    vhg_key=vhg.key,
    use_vhg=none,
    scenario=none,
    value_vhg=vhg.value_vhg,
    value_diff=vhg.value_diff,
    b_full=true
) }}
{% endfor %}
