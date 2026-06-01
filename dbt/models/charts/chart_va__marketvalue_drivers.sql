{% set chart_jaar = var("jaar") %}
{% set prior_year = var("jaar") - 1 %}
{% set top_n = 7 %}

-- Determine Basis/Full per Corporatie based on Validatie disconteringsvoet
with waarderingstype as (
    select
        "Corporatie",
        case
            when sum(
                case when "stepName" = 'Validatie disconteringsvoet'
                    then "valuePerClassificationAndModel.TOTAL.RESIDENTIAL.relativeDifferenceWithFirstStep"
                    else 0
                end
            ) = 0 then 'Full'
            else 'Basis'
        end as "Waarderingstype"
    from {{ source("tms", "tms_verschillenanalyse_marketvalue") }}
    where "Jaar" = {{ chart_jaar }}
    group by "Corporatie"
),

-- Base value per Waarderingstype (Waarde {prior_year} row, group=None)
base_values as (
    select
        w."Waarderingstype",
        sum(v.{{ residential_delta_with_previous_step() }}) as base_value
    from {{ source("tms", "tms_verschillenanalyse_marketvalue") }} v
    join waarderingstype w on v."Corporatie" = w."Corporatie"
    where v."Jaar" = {{ chart_jaar }}
      and coalesce(v."groupName", 'None') = 'None'
      and v."stepName" = 'Waarde {{ prior_year }}'
    group by w."Waarderingstype"
),

-- Map (groupName, stepName) to cleaned step name, sum deltas
effects as (
    select
        w."Waarderingstype",
        case
            when coalesce(v."groupName", 'None') in ('None', 'Voorraadmutaties') then null
            when v."groupName" = 'Methodische wijzigingen' then 'Methodische wijzigingen'
            when v."stepName" = 'Validatie disconteringsvoet' then 'Disconteringsvoet'
            when v."stepName" = 'Validatie markthuur' then 'Markthuur'
            when v."stepName" in ('Leegwaarde', 'WOZ-waarde', 'Historische leegwaardestijging', 'Leegwaardestijging')
                then 'Leeg- en WOZ-waarde(-stijging)'
            when v."stepName" = 'Macro-economische parameters' then 'Indexeringen'
            when v."stepName" = 'Instandhoudings- en mutatieonderhoud' then 'Onderhoud'
            else v."stepName"
        end as "Stap",
        sum(v.{{ residential_delta_with_previous_step() }}) as delta_sum
    from {{ source("tms", "tms_verschillenanalyse_marketvalue") }} v
    join waarderingstype w on v."Corporatie" = w."Corporatie"
    where v."Jaar" = {{ chart_jaar }}
    group by w."Waarderingstype", "Stap"
    having "Stap" is not null
),

-- Divide by base value to get fractional effect
effects_pct as (
    select
        e."Waarderingstype",
        e."Stap",
        e.delta_sum / nullif(b.base_value, 0) as "Mutatie"
    from effects e
    join base_values b on e."Waarderingstype" = b."Waarderingstype"
),

-- Rank effects per Waarderingstype by absolute magnitude
ranked_per_type as (
    select
        "Waarderingstype",
        "Stap",
        "Mutatie",
        row_number() over (partition by "Waarderingstype" order by abs("Mutatie") desc) as rn
    from effects_pct
),

-- Top-N step names per type (union of sets)
selected_steps as (
    select distinct "Stap"
    from ranked_per_type
    where rn <= {{ top_n }}
),

-- Order selected steps by mean absolute effect across all types
step_order as (
    select
        s."Stap",
        row_number() over (order by avg(abs(e."Mutatie")) desc) as step_order
    from selected_steps s
    join effects_pct e on s."Stap" = e."Stap"
    group by s."Stap"
),

-- Get effects for selected steps; zero Exit yield for Basis
ordered_effects as (
    select
        e."Waarderingstype",
        so."Stap",
        case
            when so."Stap" = 'Exit yield' and e."Waarderingstype" = 'Basis' then 0
            else e."Mutatie"
        end as "Mutatie",
        so.step_order
    from effects_pct e
    join step_order so on e."Stap" = so."Stap"
),

-- Calculate cumulative sum
with_cumulative as (
    select
        "Waarderingstype",
        "Stap",
        "Mutatie",
        sum("Mutatie") over (partition by "Waarderingstype" order by step_order) as "Cumulatief",
        step_order
    from ordered_effects
),

-- Total % change per Waarderingstype from housing value development
waardestijging as (
    select "Waarderingstype", "% Mutatie" as total_pct
    from {{ ref("chart_waardeontwikkeling_won__current_t0") }}
    where "Waarderingstype" in ('Basis', 'Full')
),

-- Last cumulative value per type
last_cumul as (
    select "Waarderingstype", "Cumulatief"
    from with_cumulative
    qualify row_number() over (partition by "Waarderingstype" order by step_order desc) = 1
),

-- "Overige mutaties" remainder row
overige as (
    select
        ws."Waarderingstype",
        'Overige mutaties' as "Stap",
        ws.total_pct - lc."Cumulatief" as "Mutatie",
        ws.total_pct as "Cumulatief",
        (select max(step_order) + 1 from with_cumulative) as step_order
    from waardestijging ws
    join last_cumul lc on ws."Waarderingstype" = lc."Waarderingstype"
),

final as (
    select "Waarderingstype", "Stap", "Mutatie", "Cumulatief", step_order
    from with_cumulative
    union all
    select "Waarderingstype", "Stap", "Mutatie", "Cumulatief", step_order
    from overige
)

{{ va_driver_output('"Waarderingstype", step_order') }}
