{#
    All 10 marktwaarde VHG indexatiegebied bar-chart datasets in one model.
    Each VHG produces Basis + Full + Full(handboek) rows per Indexatiegebied.
    Python filters by vhg_key for each chart.
#}

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

{% for vhg in vhgs %}
{% if not loop.first %}union all{% endif %}
{{ vhg_indexatiegebied_mw(
    vhg_key=vhg.key,
    use_vhg=vhg.use_vhg,
    scenario=vhg.scenario,
    value_vhg=vhg.value_vhg,
    value_basic=vhg.value_basic,
    value_diff=vhg.value_diff
) }}
{% endfor %}
