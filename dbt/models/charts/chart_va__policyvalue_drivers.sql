{% set chart_jaar = var("jaar") %}
{% set prior_year = var("jaar") - 1 %}
{% set top_n = 5 %}

-- Beleidswaarde: single Waarderingstype, no Basis/Full split
with base_values as (
    select
        'Beleidswaarde' as "Waarderingstype",
        sum({{ residential_delta_with_previous_step() }}) as base_value
    from {{ source("tms", "tms_verschillenanalyse_policyvalue") }}
    where "Jaar" = {{ chart_jaar }}
      and coalesce("groupName", 'None') = 'None'
      and "stepName" = 'Waarde {{ prior_year }}'
),

-- Map (groupName, stepName) to cleaned step name, sum deltas
effects as (
    select
        'Beleidswaarde' as "Waarderingstype",
        case
            when coalesce("groupName", 'None') in ('None', 'Voorraadmutaties') then null
            when "groupName" = 'Methodische wijzigingen' then 'Methodische wijzigingen'
            when "stepName" = 'Validatie disconteringsvoet' then 'Disconteringsvoet'
            when "stepName" in ('WOZ-waarde', 'Leegwaardestijging') then 'WOZ-waarde(-stijging)'
            when "stepName" = 'Macro-economische parameters' then 'Indexeringen'
            when "stepName" = 'Instandhoudings- en mutatieonderhoud' then 'Onderhoud'
            else "stepName"
        end as "Stap",
        sum({{ residential_delta_with_previous_step() }}) as delta_sum
    from {{ source("tms", "tms_verschillenanalyse_policyvalue") }}
    where "Jaar" = {{ chart_jaar }}
    group by "Stap"
    having "Stap" is not null
),

-- Divide by base value to get fractional effect
effects_pct as (
    select
        e."Waarderingstype",
        e."Stap",
        e.delta_sum / nullif(b.base_value, 0) as "Mutatie"
    from effects e
    cross join base_values b
),

-- Rank effects by absolute magnitude
ranked as (
    select
        "Waarderingstype",
        "Stap",
        "Mutatie",
        row_number() over (order by abs("Mutatie") desc) as rn
    from effects_pct
),

-- Top-N steps
selected_steps as (
    select "Stap", rn as step_order
    from ranked
    where rn <= {{ top_n }}
),

-- Get effects for selected steps with ordering
ordered_effects as (
    select
        e."Waarderingstype",
        so."Stap",
        e."Mutatie",
        so.step_order
    from effects_pct e
    join selected_steps so on e."Stap" = so."Stap"
),

-- Calculate cumulative sum
with_cumulative as (
    select
        "Waarderingstype",
        "Stap",
        "Mutatie",
        sum("Mutatie") over (order by step_order) as "Cumulatief",
        step_order
    from ordered_effects
),

-- Total % change for Beleidswaarde from housing value development
waardestijging as (
    select "% Mutatie" as total_pct
    from {{ ref("chart_waardeontwikkeling_won__current_t0") }}
    where "Waarderingstype" = 'Beleidswaarde'
),

-- Last cumulative value
last_cumul as (
    select "Cumulatief"
    from with_cumulative
    qualify row_number() over (order by step_order desc) = 1
),

-- "Overige mutaties" remainder row
overige as (
    select
        'Beleidswaarde' as "Waarderingstype",
        'Overige mutaties' as "Stap",
        ws.total_pct - lc."Cumulatief" as "Mutatie",
        ws.total_pct as "Cumulatief",
        (select max(step_order) + 1 from with_cumulative) as step_order
    from waardestijging ws
    cross join last_cumul lc
),

final as (
    select "Waarderingstype", "Stap", "Mutatie", "Cumulatief", step_order
    from with_cumulative
    union all
    select "Waarderingstype", "Stap", "Mutatie", "Cumulatief", step_order
    from overige
)

{{ va_driver_output("step_order") }}
