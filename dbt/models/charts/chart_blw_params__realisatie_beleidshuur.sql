{% set chart_jaar = var("jaar") %}
{% set prior_year = chart_jaar - 1 %}

-- Item 17: Realisatie beleidshuur
-- Filter: EGW/MGW met gemeten huurstijging < 0% of > 5% (mutaties)
-- Per company: % woningen waar netto huur (VGG 2025) exact gelijk aan beleidshuur (BLW params 2025)

with vgg_current as (
    select
        "Corporatie",
        "VHE-nummer",
        TRY_CAST("Netto huur" AS DOUBLE) as netto_huur_current
    from {{ source("tms", "vastgoedgegevens_vhe_gegevens") }}
    where "Jaar" = {{ chart_jaar }}
      and "Handboektype" in ('EGW', 'MGW')
      and TRY_CAST("Netto huur" AS DOUBLE) > 0
),

vgg_prior as (
    select
        "Corporatie",
        "VHE-nummer",
        TRY_CAST("Netto huur" AS DOUBLE) as netto_huur_prior
    from {{ source("tms", "vastgoedgegevens_vhe_gegevens") }}
    where "Jaar" = {{ prior_year }}
      and "Handboektype" in ('EGW', 'MGW')
      and TRY_CAST("Netto huur" AS DOUBLE) > 0
),

pvp_current as (
    select
        "Corporatie",
        "VHE-nr",
        TRY_CAST("Beleidshuur" AS DOUBLE) as beleidshuur
    from {{ source("tms", "policy_value_parameters") }}
    where "Jaar" = {{ chart_jaar }}
      and "Beleidshuur" is not null
),

joined as (
    select
        vc."Corporatie",
        vc."VHE-nummer",
        vc.netto_huur_current,
        vp.netto_huur_prior,
        (vc.netto_huur_current - vp.netto_huur_prior) / vp.netto_huur_prior as gemeten_huurstijging,
        pvp.beleidshuur,
        case when round(vc.netto_huur_current, 2) = round(pvp.beleidshuur, 2) then 1 else 0 end as is_gelijk_beleidshuur
    from vgg_current vc
    inner join vgg_prior vp
        on vc."Corporatie" = vp."Corporatie"
        and vc."VHE-nummer" = vp."VHE-nummer"
    inner join pvp_current pvp
        on vc."Corporatie" = pvp."Corporatie"
        and vc."VHE-nummer" = pvp."VHE-nr"
)

select
    "Corporatie",
    sum(is_gelijk_beleidshuur) * 1.0 / count(*) as "Percentage netto huur = beleidshuur",
    count(*) as "Aantal VHE"
from joined
where gemeten_huurstijging < 0.0
   or gemeten_huurstijging > 0.05
group by "Corporatie"
