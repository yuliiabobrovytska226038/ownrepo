{#
    All 4 beleidswaarde VHG indexatiegebied bar-chart datasets in one model.
    Python filters by vhg_key for each chart.
#}

{% set vhgs = [
    {"key": "sh", "value_vhg": "Beleidshuur", "value_diff": "Relatief"},
    {"key": "bb", "value_vhg": "Beleidsbeheer", "value_diff": "Relatief"},
    {"key": "bo", "value_vhg": "Beleidsonderhoud", "value_diff": "Relatief"},
    {"key": "dv", "value_vhg": "DV doorexploiteren", "value_diff": "Relatief"},
] %}

{% for vhg in vhgs %}
{% if not loop.first %}union all{% endif %}
{{ vhg_indexatiegebied_blw(
    vhg_key=vhg.key,
    value_vhg=vhg.value_vhg,
    value_diff=vhg.value_diff
) }}
{% endfor %}
