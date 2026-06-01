{% set chart_jaar = var("jaar") %}
{% set prior_year = chart_jaar - 1 %}

-- Item 18: Realisatie MJOB
-- Per company: vergelijking verwachte beleidsonderhoud (jaar 1 uit BLW params vorig jaar)
-- met daadwerkelijk toegepaste beleidsonderhoud (BLW params huidig jaar)

with pvp_prior as (
    select
        "Corporatie",
        "VHE-nr",
        TRY_CAST("Beleidsonderhoud jaar 1" AS DOUBLE) as bo_verwacht
    from {{ source("tms", "policy_value_parameters") }}
    where "Jaar" = {{ prior_year }}
      and TRY_CAST("Beleidsonderhoud jaar 1" AS DOUBLE) is not null
),

pvp_current as (
    select
        "Corporatie",
        "VHE-nr",
        TRY_CAST("Beleidsonderhoud jaar 1" AS DOUBLE) as bo_actueel
    from {{ source("tms", "policy_value_parameters") }}
    where "Jaar" = {{ chart_jaar }}
      and TRY_CAST("Beleidsonderhoud jaar 1" AS DOUBLE) is not null
),

joined as (
    select
        pc."Corporatie",
        pc."VHE-nr",
        pp.bo_verwacht,
        pc.bo_actueel,
        case
            when pp.bo_verwacht > 0
            then (pc.bo_actueel - pp.bo_verwacht) / pp.bo_verwacht
            else null
        end as verschil_ratio
    from pvp_current pc
    inner join pvp_prior pp
        on pc."Corporatie" = pp."Corporatie"
        and pc."VHE-nr" = pp."VHE-nr"
    where pp.bo_verwacht > 0
)

select
    "Corporatie",
    median(verschil_ratio) as "Mediaan verschil realisatie MJOB",
    avg(verschil_ratio) as "Gem verschil realisatie MJOB",
    count(*) as "Aantal VHE"
from joined
group by "Corporatie"
