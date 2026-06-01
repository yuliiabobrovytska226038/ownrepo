{#
    All 4 beleidswaarde VHG year-over-year boxplot datasets.
    Python filters by vhg_key for each chart.
#}

{% set vhgs = [
    {"key": "sh", "value_vhg": "Beleidshuur"},
    {"key": "bb", "value_vhg": "Beleidsbeheer"},
    {"key": "bo", "value_vhg": "Beleidsonderhoud"},
    {"key": "dv", "value_vhg": "DV doorexploiteren"},
] %}

{% for vhg in vhgs %}
{% if not loop.first %}union all{% endif %}
{{ vhg_blw_yoy(
    vhg_key=vhg.key,
    value_vhg=vhg.value_vhg
) }}
{% endfor %}
