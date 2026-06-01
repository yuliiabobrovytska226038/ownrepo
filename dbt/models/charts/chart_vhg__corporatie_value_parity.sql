{% set prior_year = var("jaar") - 1 %}
{% set value_ctes = [
    "mw_dv_de",
    "mw_dv_up",
    "mw_lw",
    "mw_mh",
    "mw_oh_de",
    "mw_oh_up",
    "mw_mu_de",
    "mw_mu_up",
    "mw_ey_de",
    "mw_ey_up",
    "bw_sh",
    "bw_bb",
    "bw_bo",
    "bw_dv",
] %}

with base as (
    select
        *,
        case when "Bron Onderhoud (OH)" = 'VTW' then 'Eigen invoer' else "Bron Onderhoud (OH)" end as bron_onderhoud_oh,
        case when "Bron Onderhoud (OH)_{{ prior_year }}" = 'VTW' then 'Eigen invoer' else "Bron Onderhoud (OH)_{{ prior_year }}" end as bron_onderhoud_oh_prior
    from {{ ref("int_chart_ontwikkeling_woningen_egw_mgw") }}
),

mw_dv_de as (
    select
        'Marktwaarde' as "Waarde",
        'dv_de' as "VhgKey",
        'DV doorexploiteren' as "ValueColumn",
        corporatie as "Corporatie",
        waarderingstype as "Waarderingstype",
        round(avg("DV doorexploiteren") - avg("DV doorexploiteren_{{ prior_year }}"), 6) as "MetricValue"
    from base
    where marktwaarde is not null
      and {{ vhg_current_and_prior_eigen_invoer("Bron disconteringsvoet (DV)") }}
      and {{ vhg_scenario_allowed("U") }}
      and {{ vhg_scenario_allowed("U", "_" ~ prior_year) }}
      and corporatie not in ('Compaen', 'Domijn', 'WelWonen')
    group by corporatie, waarderingstype
),

mw_dv_up as (
    select
        'Marktwaarde' as "Waarde",
        'dv_up' as "VhgKey",
        'DV uitponden' as "ValueColumn",
        corporatie as "Corporatie",
        waarderingstype as "Waarderingstype",
        round(avg("DV uitponden") - avg("DV uitponden_{{ prior_year }}"), 6) as "MetricValue"
    from base
    where marktwaarde is not null
      and {{ vhg_current_and_prior_eigen_invoer("Bron disconteringsvoet (DV)") }}
      and {{ vhg_scenario_allowed("D") }}
      and {{ vhg_scenario_allowed("D", "_" ~ prior_year) }}
      and corporatie not in ('3B Wonen', 'SSHN', 'Woonbedrijf SWS.Hhvl', 'Idealis', 'WelWonen')
    group by corporatie, waarderingstype
),

mw_lw as (
    select
        'Marktwaarde' as "Waarde",
        'lw' as "VhgKey",
        'LW waarde' as "ValueColumn",
        corporatie as "Corporatie",
        waarderingstype as "Waarderingstype",
        round((avg("LW waarde") - avg("LW waarde_{{ prior_year }}")) / nullif(avg("LW waarde_{{ prior_year }}"), 0), 6) as "MetricValue"
    from base
    where marktwaarde is not null
      and {{ vhg_current_and_prior_eigen_invoer("Bron leegwaarde (LW)") }}
      and {{ vhg_scenario_allowed("D") }}
      and {{ vhg_scenario_allowed("D", "_" ~ prior_year) }}
      and corporatie not in ('WelWonen', 'JOOST')
    group by corporatie, waarderingstype
),

mw_mh as (
    select
        'Marktwaarde' as "Waarde",
        'mh' as "VhgKey",
        'MH waarde' as "ValueColumn",
        corporatie as "Corporatie",
        waarderingstype as "Waarderingstype",
        round((avg("MH waarde") - avg("MH waarde_{{ prior_year }}")) / nullif(avg("MH waarde_{{ prior_year }}"), 0), 6) as "MetricValue"
    from base
    where marktwaarde is not null
      and {{ vhg_current_and_prior_eigen_invoer("Bron markthuur (MH)") }}
      and {{ vhg_scenario_allowed("U") }}
      and {{ vhg_scenario_allowed("U", "_" ~ prior_year) }}
      and corporatie not in ('Uwoon', 'Actium', 'JOOST', 'DUWO')
    group by corporatie, waarderingstype
),

mw_oh_de as (
    select
        'Marktwaarde' as "Waarde",
        'oh_de' as "VhgKey",
        'OH doorexploiteren' as "ValueColumn",
        corporatie as "Corporatie",
        waarderingstype as "Waarderingstype",
        round((avg("OH doorexploiteren") - avg("OH doorexploiteren_{{ prior_year }}")) / nullif(avg("OH doorexploiteren_{{ prior_year }}"), 0), 6) as "MetricValue"
    from base
    where marktwaarde is not null
      and bron_onderhoud_oh = 'Eigen invoer'
      and bron_onderhoud_oh_prior = 'Eigen invoer'
      and {{ vhg_scenario_allowed("U") }}
      and {{ vhg_scenario_allowed("U", "_" ~ prior_year) }}
      and corporatie not in ('Compaen', 'Woonborg', 'SSHN', 'Parteon', 'Wassenaarsche Bouwstichting', 'Woonveste', 'Woongoed Middelburg', 'KleurrijkWonen', 'Trudo', 'BrabantWonen', 'DeZaligheden', 'WonenBreburg', 'Uwoon', 'Woonbedrijf SWS.Hhvl', 'Rochdale')
    group by corporatie, waarderingstype
),

mw_oh_up as (
    select
        'Marktwaarde' as "Waarde",
        'oh_up' as "VhgKey",
        'OH uitponden' as "ValueColumn",
        corporatie as "Corporatie",
        waarderingstype as "Waarderingstype",
        round((avg("OH uitponden") - avg("OH uitponden_{{ prior_year }}")) / nullif(avg("OH uitponden_{{ prior_year }}"), 0), 6) as "MetricValue"
    from base
    where marktwaarde is not null
      and bron_onderhoud_oh = 'Eigen invoer'
      and bron_onderhoud_oh_prior = 'Eigen invoer'
      and {{ vhg_scenario_allowed("D") }}
      and {{ vhg_scenario_allowed("D", "_" ~ prior_year) }}
      and corporatie not in ('Compaen', 'Uwoon', 'Woonborg', 'SSHN', 'Woonbedrijf SWS.Hhvl', 'Rochdale', 'Wassenaarsche Bouwstichting', 'Woonveste', 'Woongoed Middelburg', 'Parteon')
    group by corporatie, waarderingstype
),

mw_mu_de as (
    select
        'Marktwaarde' as "Waarde",
        'mu_de' as "VhgKey",
        'MD waarde' as "ValueColumn",
        corporatie as "Corporatie",
        waarderingstype as "Waarderingstype",
        round(avg("MD waarde") - avg("MD waarde_{{ prior_year }}"), 6) as "MetricValue"
    from base
    where marktwaarde is not null
      and {{ vhg_current_and_prior_eigen_invoer("Bron mutatiegraad doorexploiteren (MD)") }}
      and {{ vhg_scenario_allowed("U") }}
      and {{ vhg_scenario_allowed("U", "_" ~ prior_year) }}
    group by corporatie, waarderingstype
),

mw_mu_up as (
    select
        'Marktwaarde' as "Waarde",
        'mu_up' as "VhgKey",
        'MU jaar 1-15' as "ValueColumn",
        corporatie as "Corporatie",
        waarderingstype as "Waarderingstype",
        round(avg("MU jaar 1-15") - avg("MU jaar 1-15_{{ prior_year }}"), 6) as "MetricValue"
    from base
    where marktwaarde is not null
      and {{ vhg_current_and_prior_eigen_invoer("Bron mutatiegraad uitponden (MU)") }}
      and {{ vhg_scenario_allowed("D") }}
      and {{ vhg_scenario_allowed("D", "_" ~ prior_year) }}
    group by corporatie, waarderingstype
),

mw_ey_de as (
    select
        'Marktwaarde' as "Waarde",
        'ey_de' as "VhgKey",
        'EY doorexploiteren' as "ValueColumn",
        corporatie as "Corporatie",
        waarderingstype as "Waarderingstype",
        round(avg("EY doorexploiteren") - avg("EY doorexploiteren_{{ prior_year }}"), 6) as "MetricValue"
    from base
    where marktwaarde is not null
      and {{ vhg_current_and_prior_eigen_invoer("Bron exit yield (EY)") }}
      and {{ vhg_scenario_allowed("U") }}
      and {{ vhg_scenario_allowed("U", "_" ~ prior_year) }}
    group by corporatie, waarderingstype
),

mw_ey_up as (
    select
        'Marktwaarde' as "Waarde",
        'ey_up' as "VhgKey",
        'EY uitponden' as "ValueColumn",
        corporatie as "Corporatie",
        waarderingstype as "Waarderingstype",
        round(avg("EY uitponden") - avg("EY uitponden_{{ prior_year }}"), 6) as "MetricValue"
    from base
    where marktwaarde is not null
      and {{ vhg_current_and_prior_eigen_invoer("Bron exit yield (EY)") }}
      and {{ vhg_scenario_allowed("D") }}
      and {{ vhg_scenario_allowed("D", "_" ~ prior_year) }}
    group by corporatie, waarderingstype
),

bw_sh as (
    select
        'Beleidswaarde' as "Waarde",
        'sh' as "VhgKey",
        'Beleidshuur' as "ValueColumn",
        corporatie as "Corporatie",
        null::varchar as "Waarderingstype",
        round((avg(beleidshuur) - avg("Beleidshuur_{{ prior_year }}")) / nullif(avg("Beleidshuur_{{ prior_year }}"), 0), 6) as "MetricValue"
    from base
    where beleidswaarde is not null
    group by corporatie
),

bw_bb as (
    select
        'Beleidswaarde' as "Waarde",
        'bb' as "VhgKey",
        'Beleidsbeheer' as "ValueColumn",
        corporatie as "Corporatie",
        null::varchar as "Waarderingstype",
        round((avg(beleidsbeheer_bp) - avg("Beleidsbeheer_{{ prior_year }}")) / nullif(avg("Beleidsbeheer_{{ prior_year }}"), 0), 6) as "MetricValue"
    from base
    where beleidswaarde is not null
    group by corporatie
),

bw_bo as (
    select
        'Beleidswaarde' as "Waarde",
        'bo' as "VhgKey",
        'Beleidsonderhoud' as "ValueColumn",
        corporatie as "Corporatie",
        null::varchar as "Waarderingstype",
        round((avg(beleidsonderhoud) - avg("Beleidsonderhoud_{{ prior_year }}")) / nullif(avg("Beleidsonderhoud_{{ prior_year }}"), 0), 6) as "MetricValue"
    from base
    where beleidswaarde is not null
    group by corporatie
),

bw_dv as (
    select
        'Beleidswaarde' as "Waarde",
        'dv' as "VhgKey",
        'DV doorexploiteren' as "ValueColumn",
        corporatie as "Corporatie",
        null::varchar as "Waarderingstype",
        round((avg("DV doorexploiteren") - avg("DV doorexploiteren_{{ prior_year }}")) / nullif(avg("DV doorexploiteren_{{ prior_year }}"), 0), 6) as "MetricValue"
    from base
    where beleidswaarde is not null
      and corporatie not in ('Compaen', 'Domijn', 'WelWonen')
    group by corporatie
)

{% for value_cte in value_ctes %}
{% if not loop.first %}union all{% endif %}
select * from {{ value_cte }}
{% endfor %}
