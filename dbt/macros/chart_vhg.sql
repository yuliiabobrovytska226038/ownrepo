{#
    Macros for generating VHG (vrijheidsgraad) chart data.
    Each macro produces filtered + aggregated data for one VHG configuration,
    parameterised by the columns and diff-type from chart_constants.py.
#}


{% macro vhg_filter_where(use_vhg=none, scenario=none, b_full=true, value_vhg=none, value_basic=none) %}
{#
    Generate WHERE clauses for VHG filtering.
    Mirrors _filter_df_vhg() from Python chart_vhg_ontwikkeling.py.
    Applies: use_vhg matching, scenario filtering, outlier exclusions, sentinel rate filtering.
#}
{% set prior_year = var("jaar") - 1 %}
{% set rate_columns = [
    'EY doorexploiteren', 'EY uitponden',
    'EY doorexploiteren handboek', 'EY uitponden handboek',
    'MD waarde', 'MD handboek waarde',
    'MU jaar 1-15', 'MU jaar 1-15 handboek'
] %}

    {# use_vhg matching: current year = prior year #}
    {% if use_vhg %}
    and "{{ use_vhg }}" = "{{ use_vhg }}_{{ prior_year }}"
        {% if b_full %}
    and "{{ use_vhg }}" = 'Eigen invoer'
    and "{{ use_vhg }}_{{ prior_year }}" = 'Eigen invoer'
        {% endif %}
    {% endif %}

    {# Scenario filtering #}
    {% if scenario == 'Uitponden' %}
    and not ("Bron scenario (SC)" = 'Eigen invoer' and "SC waarde" = 'D')
    and not ("Bron scenario (SC)_{{ prior_year }}" = 'Eigen invoer' and "SC waarde_{{ prior_year }}" = 'D')
    {% elif scenario == 'Doorexploiteren' %}
    and not ("Bron scenario (SC)" = 'Eigen invoer' and "SC waarde" = 'U')
    and not ("Bron scenario (SC)_{{ prior_year }}" = 'Eigen invoer' and "SC waarde_{{ prior_year }}" = 'U')
    {% endif %}

    {# Outlier exclusions from seed table #}
    {% if value_vhg %}
    and corporatie not in (
        select corporatie
        from {{ ref("chart_vhg__outlier_exclusions") }}
        where value_vhg = '{{ value_vhg }}'
          and jaar = {{ var("jaar") }}
    )
    {% endif %}

    {# Sentinel rate filtering: values > 1.0 are TMS placeholders #}
    {% if value_vhg in rate_columns %}
    and ("{{ value_vhg }}" is null or "{{ value_vhg }}" <= 1.0)
    and ("{{ value_vhg }}_{{ prior_year }}" is null or "{{ value_vhg }}_{{ prior_year }}" <= 1.0)
    {% endif %}
    {% if value_basic and value_basic in rate_columns %}
    and ("{{ value_basic }}" is null or "{{ value_basic }}" <= 1.0)
    and ("{{ value_basic }}_{{ prior_year }}" is null or "{{ value_basic }}_{{ prior_year }}" <= 1.0)
    {% endif %}

{% endmacro %}


{% macro vhg_scenario_allowed(scenario_value, suffix="") -%}
("Bron scenario (SC){{ suffix }}" is null or "Bron scenario (SC){{ suffix }}" <> 'Eigen invoer' or "SC waarde{{ suffix }}" is null or "SC waarde{{ suffix }}" <> '{{ scenario_value }}')
{%- endmacro %}


{% macro vhg_current_and_prior_eigen_invoer(source_column) -%}
"{{ source_column }}" = 'Eigen invoer'
and "{{ source_column }}_{{ var("jaar") - 1 }}" = 'Eigen invoer'
{%- endmacro %}


{% macro vhg_indexatiegebied_mw(vhg_key, use_vhg, scenario, value_vhg, value_basic, value_diff) %}
{#
    Generate indexatiegebied bar-chart data for one MW VHG.
    Mirrors _t_indexatiegebied_value() — pivot by Indexatiegebied × Waarderingstype.
    Returns long-format: vhg_key, Indexatiegebied, Waarderingstype, value.
    Uses b_full=False for indexatiegebied charts.
#}
{% set prior_year = var("jaar") - 1 %}

    select
        '{{ vhg_key }}' as vhg_key,
        aggregated."Indexatiegebied",
        value_rows."Waarderingstype",
        value_rows.value
    from (
        select
            "Indexatiegebied",
            avg(case when "Waarderingstype" = 'Basis' then "{{ value_vhg }}" end) as basis_current,
            avg(case when "Waarderingstype" = 'Basis' then "{{ value_vhg }}_{{ prior_year }}" end) as basis_prior,
            avg(case when "Waarderingstype" = 'Full' then "{{ value_vhg }}" end) as full_current,
            avg(case when "Waarderingstype" = 'Full' then "{{ value_vhg }}_{{ prior_year }}" end) as full_prior,
            avg(case when "Waarderingstype" = 'Full' then "{{ value_basic }}" end) as full_basic_current,
            avg(case when "Waarderingstype" = 'Full' then "{{ value_basic }}_{{ prior_year }}" end) as full_basic_prior
        from {{ ref("chart_vhg__mw_prepared") }}
        where 1=1
        {{ vhg_filter_where(use_vhg=use_vhg, scenario=scenario, b_full=false, value_vhg=value_vhg, value_basic=value_basic) }}
        group by "Indexatiegebied"
    ) as aggregated
    cross join lateral (
        values
            ('Basis',
        {% if value_diff == 'Absoluut' %}
                basis_current - basis_prior
        {% else %}
                (basis_current - basis_prior) / nullif(basis_prior, 0)
        {% endif %}
            ),
            ('Full',
        {% if value_diff == 'Absoluut' %}
                full_current - full_prior
        {% else %}
                (full_current - full_prior) / nullif(full_prior, 0)
        {% endif %}
            ),
            ('Full (handboek)',
        {% if value_diff == 'Absoluut' %}
                full_basic_current - full_basic_prior
        {% else %}
                (full_basic_current - full_basic_prior) / nullif(full_basic_prior, 0)
        {% endif %}
            )
    ) as value_rows("Waarderingstype", value)

{% endmacro %}


{% macro vhg_indexatiegebied_blw(vhg_key, value_vhg, value_diff) %}
{#
    Generate indexatiegebied bar-chart data for one BLW VHG.
    Mirrors _t_indexatiegebied_value_blw() — single Waarderingstype='Beleidswaarde'.
#}
{% set prior_year = var("jaar") - 1 %}

    select
        '{{ vhg_key }}' as vhg_key,
        "Indexatiegebied",
        'Beleidswaarde' as "Waarderingstype",
        {% if value_diff == 'Relatief' %}
        (avg("{{ value_vhg }}") - avg("{{ value_vhg }}_{{ prior_year }}"))
        / nullif(avg("{{ value_vhg }}_{{ prior_year }}"), 0)
        {% else %}
        avg("{{ value_vhg }}") - avg("{{ value_vhg }}_{{ prior_year }}")
        {% endif %}
        as value
    from {{ ref("chart_vhg__blw_prepared") }}
    where 1=1
    {{ vhg_filter_where(value_vhg=value_vhg) }}
    group by "Indexatiegebied"

{% endmacro %}


{% macro vhg_corporatie_value_pair(source_ref, vhg_key, use_vhg, scenario, value_vhg, value_diff, b_full=true) %}
{#
    Generate both niveau and aanpassing corporatie boxplot rows for one VHG.
    Shares one filtered aggregate for both calc_type outputs.
#}
{% set prior_year = var("jaar") - 1 %}

    select
        '{{ vhg_key }}' as vhg_key,
        aggregated.corporatie as "Corporatie",
        aggregated."Waarderingstype",
        value_rows.value,
        value_rows.calc_type
    from (
        select
            corporatie,
            "Waarderingstype",
            avg("{{ value_vhg }}") as niveau_value,
        {% if value_diff == 'Absoluut' %}
            avg("{{ value_vhg }}") - avg("{{ value_vhg }}_{{ prior_year }}") as aanpassing_value
        {% else %}
            (avg("{{ value_vhg }}") - avg("{{ value_vhg }}_{{ prior_year }}"))
            / nullif(avg("{{ value_vhg }}_{{ prior_year }}"), 0) as aanpassing_value
        {% endif %}
        from {{ ref(source_ref) }}
        where 1=1
        {{ vhg_filter_where(use_vhg=use_vhg, scenario=scenario, b_full=b_full, value_vhg=value_vhg) }}
        group by corporatie, "Waarderingstype"
    ) as aggregated
    cross join lateral (
        values
            (niveau_value, 'niveau'),
            (aanpassing_value, 'aanpassing')
    ) as value_rows(value, calc_type)

{% endmacro %}


{% macro vhg_corporatie_boxplot_vhg(vhg_key, use_vhg, scenario, value_vhg, value_basic, value_diff) %}
{#
    Generate Basis/Full/Full(handboek) boxplot data for one MW VHG.
    Mirrors _t_vhg_corporatie() — pivot + melt into long-format.
    Returns: vhg_key, Waarderingstype, Corporatie, Vrijheidsgraad, Waarde.
#}
{% set prior_year = var("jaar") - 1 %}

    select
        '{{ vhg_key }}' as vhg_key,
        value_rows."Waarderingstype",
        aggregated.corporatie as "Corporatie",
        value_rows."Vrijheidsgraad",
        value_rows."Waarde"
    from (
        select
            corporatie,
            count(*) filter (where "Waarderingstype" = 'Full') as full_count,
            count(*) filter (where "Waarderingstype" = 'Basis') as basis_count,
            avg("{{ value_vhg }}") filter (where "Waarderingstype" = 'Full') as full_current,
            avg("{{ value_vhg }}_{{ prior_year }}") filter (where "Waarderingstype" = 'Full') as full_prior,
            avg("{{ value_basic }}") filter (where "Waarderingstype" = 'Full') as full_basic_current,
            avg("{{ value_basic }}_{{ prior_year }}") filter (where "Waarderingstype" = 'Full') as full_basic_prior,
            avg("{{ value_vhg }}") filter (where "Waarderingstype" = 'Basis') as basis_current,
            avg("{{ value_vhg }}_{{ prior_year }}") filter (where "Waarderingstype" = 'Basis') as basis_prior,
        {% if value_diff == 'Absoluut' %}
            avg("{{ value_vhg }}") filter (where "Waarderingstype" = 'Full')
            - avg("{{ value_vhg }}_{{ prior_year }}") filter (where "Waarderingstype" = 'Full') as full_diff,
            avg("{{ value_basic }}") filter (where "Waarderingstype" = 'Full')
            - avg("{{ value_basic }}_{{ prior_year }}") filter (where "Waarderingstype" = 'Full') as full_basic_diff,
            avg("{{ value_vhg }}") filter (where "Waarderingstype" = 'Basis')
            - avg("{{ value_vhg }}_{{ prior_year }}") filter (where "Waarderingstype" = 'Basis') as basis_diff
        {% else %}
            (
                avg("{{ value_vhg }}") filter (where "Waarderingstype" = 'Full')
                - avg("{{ value_vhg }}_{{ prior_year }}") filter (where "Waarderingstype" = 'Full')
            ) / nullif(avg("{{ value_vhg }}_{{ prior_year }}") filter (where "Waarderingstype" = 'Full'), 0) as full_diff,
            (
                avg("{{ value_basic }}") filter (where "Waarderingstype" = 'Full')
                - avg("{{ value_basic }}_{{ prior_year }}") filter (where "Waarderingstype" = 'Full')
            ) / nullif(avg("{{ value_basic }}_{{ prior_year }}") filter (where "Waarderingstype" = 'Full'), 0) as full_basic_diff,
            (
                avg("{{ value_vhg }}") filter (where "Waarderingstype" = 'Basis')
                - avg("{{ value_vhg }}_{{ prior_year }}") filter (where "Waarderingstype" = 'Basis')
            ) / nullif(avg("{{ value_vhg }}_{{ prior_year }}") filter (where "Waarderingstype" = 'Basis'), 0) as basis_diff
        {% endif %}
        from {{ ref("chart_vhg__mw_prepared") }}
        where 1=1
        {{ vhg_filter_where(use_vhg=use_vhg, scenario=scenario, b_full=false, value_vhg=value_vhg, value_basic=value_basic) }}
        group by corporatie
    ) as aggregated
    cross join lateral (
        values
            ('Full', '{{ value_vhg }}', full_current, full_count),
            ('Full', '{{ value_vhg }}_{{ prior_year }}', full_prior, full_count),
            ('Full (handboek)', '{{ value_basic }}', full_basic_current, full_count),
            ('Full (handboek)', '{{ value_basic }}_{{ prior_year }}', full_basic_prior, full_count),
            ('Full', 'Verschil', full_diff, full_count),
            ('Full (handboek)', 'Verschil', full_basic_diff, full_count),
            ('Basis', '{{ value_vhg }}', basis_current, basis_count),
            ('Basis', '{{ value_vhg }}_{{ prior_year }}', basis_prior, basis_count),
            ('Basis', 'Verschil', basis_diff, basis_count)
    ) as value_rows("Waarderingstype", "Vrijheidsgraad", "Waarde", row_count)
    where value_rows.row_count > 0

{% endmacro %}


{% macro vhg_blw_yoy(vhg_key, value_vhg) %}
{#
    Generate BLW year-over-year boxplot data for one VHG.
    Mirrors _g_blw_vhg_corporatie_yoy().
#}
{% set prior_year = var("jaar") - 1 %}

    select
        '{{ vhg_key }}' as vhg_key,
        aggregated.corporatie as "Corporatie",
        value_rows."Jaar",
        value_rows.value
    from (
        select
            corporatie,
            avg("{{ value_vhg }}") as current_value,
            avg("{{ value_vhg }}_{{ prior_year }}") as prior_value
        from {{ ref("chart_vhg__blw_prepared") }}
        where 1=1
        {{ vhg_filter_where(value_vhg=value_vhg) }}
        group by corporatie
    ) as aggregated
    cross join lateral (
        values
            ('{{ var("jaar") }}', current_value),
            ('{{ prior_year }}', prior_value)
    ) as value_rows("Jaar", value)

{% endmacro %}


{% macro vhg_oh_basis_full(vhg_key, value_vhg) %}
{#
    Generate OH uitponden Basis/Full comparison boxplot data.
    Mirrors _g_corporatie_oh_boxplot_basis_full().
#}
{% set prior_year = var("jaar") - 1 %}

    select
        '{{ vhg_key }}' as vhg_key,
        corporatie as "Corporatie",
        "Bron Onderhoud (OH)" as "Waarderingstype",
        avg("{{ value_vhg }}") as value
    from {{ ref("chart_vhg__mw_prepared") }}
    where "Bron Onderhoud (OH)" = "Bron Onderhoud (OH)_{{ prior_year }}"
      and not ("Bron scenario (SC)" = 'Eigen invoer' and "SC waarde" = 'D')
      and not ("Bron scenario (SC)_{{ prior_year }}" = 'Eigen invoer' and "SC waarde_{{ prior_year }}" = 'D')
    group by corporatie, "Bron Onderhoud (OH)"

{% endmacro %}
