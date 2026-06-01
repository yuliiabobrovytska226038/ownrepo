{% macro normalize_waarderingsmodel(expression) -%}
case
    when {{ expression }} = 'BOG/MOG/ZOG (harde huren)' then 'BOG/MOG/ZOG'
    else {{ expression }}
end
{%- endmacro %}


{% macro format_percent_text(expression) -%}
replace(printf('%.1f%%', {{ expression }} * 100), '.', ',')
{%- endmacro %}


{% macro waarde_types_lateral(marktwaarde_expression="marktwaarde", beleidswaarde_expression="beleidswaarde") -%}
(
    values
        ('Marktwaarde', {{ marktwaarde_expression }}),
        ('Beleidswaarde', {{ beleidswaarde_expression }})
) as waarde_types(waarde, waarde_bedrag)
{%- endmacro %}


{% macro va_driver_output(order_by) -%}
select
    "Waarderingstype",
    "Stap",
    "Mutatie",
    "Cumulatief",
    {{ format_percent_text('"Mutatie"') }} as "Text Mutatie",
    {{ format_percent_text('"Cumulatief"') }} as "Text Cumulatief"
from final
order by {{ order_by }}
{%- endmacro %}


{% macro index_mutatie(expression) -%}
{{ expression }} * 100 + 100
{%- endmacro %}


{% macro mutation_columns(current_expression, previous_expression) -%}
{{ current_expression }} - {{ previous_expression }} as "€ Mutatie",
({{ current_expression }} - {{ previous_expression }}) / nullif({{ previous_expression }}, 0) as "% Mutatie"
{%- endmacro %}


{% macro mutation_columns_when_current_nonzero(current_expression, previous_expression) -%}
{{ current_expression }} - {{ previous_expression }} as euro_mutatie,
case
    when {{ current_expression }} != 0 then ({{ current_expression }} - {{ previous_expression }}) / nullif({{ previous_expression }}, 0)
end as pct_mutatie
{%- endmacro %}


{% macro waardeontwikkeling_current_base_ctes() -%}
{% set prior_year = var("jaar") - 1 %}
normalized as (
    select
        {{ normalize_waarderingsmodel("waarderingsmodel") }} as waarderingsmodel,
        {{ normalize_waarderingsmodel('"Waarderingsmodel_' ~ prior_year ~ '"') }} as waarderingsmodel_prior,
        handboektype,
        "Handboektype_{{ prior_year }}" as handboektype_prior,
        marktwaarde,
        "Marktwaarde_{{ prior_year }}" as marktwaarde_prior,
        "VHE-nr"
    from {{ ref("dataset_ontwikkeling") }}
),
waardeontwikkeling_current_base as (
    select
        waarderingsmodel,
        waarderingsmodel_prior,
        handboektype,
        handboektype_prior,
        marktwaarde,
        marktwaarde_prior,
        "VHE-nr"
    from normalized
    where waarderingsmodel = waarderingsmodel_prior
      and waarderingsmodel <> 'Benadering'
      and marktwaarde is not null
      and marktwaarde_prior is not null
)
{%- endmacro %}


{% macro residential_delta_with_previous_step() -%}
"valuePerClassificationAndModel.TOTAL.RESIDENTIAL.deltaWithPreviousStep"
{%- endmacro %}


{% macro waarderingsmodel_order(expression) -%}
case {{ expression }}
    when 'Woningen' then 1
    when 'Parkeren' then 2
    when 'BOG/MOG/ZOG' then 3
end
{%- endmacro %}


{% macro handboektype_order(expression) -%}
case {{ expression }}
    when 'EGW' then 1
    when 'MGW' then 2
    when 'Studenteneenheid' then 3
    when 'Extramurale zorg' then 4
    when 'Parkeerplaats' then 5
    when 'Garagebox' then 6
    when 'BOG' then 7
    when 'MOG' then 8
    when 'Intramurale zorg' then 9
end
{%- endmacro %}


{% macro corrected_marktwaarde_basis() -%}
case
    when "Erfpacht (EP)" = 'Eigen invoer' and "Scenario basis" = 'EXPLOITING'
        then "Marktwaarde basis" - "EP waardecorrectie doorexploiteren" + coalesce("Erfpacht waardecorrectie doorexploiteren basis", 0)
    when "Erfpacht (EP)" = 'Eigen invoer' and "Scenario basis" = 'SELL_SCENARIO'
        then "Marktwaarde basis" - "EP waardecorrectie uitponden" + coalesce("Erfpacht waardecorrectie uitponden basis", 0)
    else "Marktwaarde basis"
end
{%- endmacro %}


{% macro basis_full_bucket(expression) -%}
case
    when {{ expression }} < -0.25 then '<-25%'
    when {{ expression }} < -0.20 then '-25% tot -20%'
    when {{ expression }} < -0.15 then '-20% tot -15%'
    when {{ expression }} < -0.10 then '-15% tot -10%'
    when {{ expression }} < -0.05 then '-10% tot -5%'
    when {{ expression }} < 0.00 then '-5% tot -0%'
    when {{ expression }} < 0.05 then '-0% tot 5%'
    when {{ expression }} < 0.10 then '5% tot 10%'
    when {{ expression }} < 0.15 then '10% tot 15%'
    when {{ expression }} < 0.20 then '15% tot 20%'
    when {{ expression }} < 0.25 then '20% tot 25%'
    else '>25%'
end
{%- endmacro %}


{% macro basis_full_bucket_order_values() -%}
values
    ('<-25%', 1),
    ('-25% tot -20%', 2),
    ('-20% tot -15%', 3),
    ('-15% tot -10%', 4),
    ('-10% tot -5%', 5),
    ('-5% tot -0%', 6),
    ('-0% tot 5%', 7),
    ('5% tot 10%', 8),
    ('10% tot 15%', 9),
    ('15% tot 20%', 10),
    ('20% tot 25%', 11),
    ('>25%', 12)
{%- endmacro %}
