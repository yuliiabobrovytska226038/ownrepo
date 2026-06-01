{% macro market_value_vhe_parameter_sources() -%}
SELECT * FROM {{ source('tms', 'market_value_parameters_vhe_param_woningen_parkeren') }}
UNION ALL BY NAME
SELECT * FROM {{ source('tms', 'market_value_parameters_vhe_param_bog_mog_zog') }}
UNION ALL BY NAME
SELECT * FROM {{ source('tms', 'market_value_parameters_vhe_param_benaderingsmethode') }}
{%- endmacro %}


{% macro market_value_complex_parameter_sources() -%}
SELECT * FROM {{ source('tms', 'market_value_parameters_complexparam_woningen_parkeren') }}
UNION ALL BY NAME
SELECT * FROM {{ source('tms', 'market_value_parameters_complexparam_bog_mog_zog') }}
{%- endmacro %}
