{% set chart_jaar = var("jaar") %}
{% set prior_year = chart_jaar - 1 %}

-- Item 16: Realisatie reguliere huurstijging
-- Per company: verschil tussen verwachte HS jaar 1 (BLW params 2024) en gemeten netto huur ontwikkeling (2024 → 2025)
-- Filter: EGW/MGW met gemeten huurstijging 0% tot 5%

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

pvp as (
    select
        "Corporatie",
        "VHE-nr",
        TRY_CAST("HS jaar 1" AS DOUBLE) as hs_jaar_1
    from {{ source("tms", "policy_value_parameters") }}
    where "Jaar" = {{ prior_year }}
      and "HS jaar 1" is not null
),

joined as (
    select
        vc."Corporatie",
        vc."VHE-nummer",
        vc.netto_huur_current,
        vp.netto_huur_prior,
        (vc.netto_huur_current - vp.netto_huur_prior) / vp.netto_huur_prior as gemeten_huurstijging,
        pvp.hs_jaar_1 as verwachte_huurstijging
    from vgg_current vc
    inner join vgg_prior vp
        on vc."Corporatie" = vp."Corporatie"
        and vc."VHE-nummer" = vp."VHE-nummer"
    inner join pvp
        on vc."Corporatie" = pvp."Corporatie"
        and vc."VHE-nummer" = pvp."VHE-nr"
)

select
    "Corporatie",
    median(gemeten_huurstijging - verwachte_huurstijging) as "Mediaan verschil realisatie",
    avg(gemeten_huurstijging - verwachte_huurstijging) as "Gem verschil realisatie",
    count(*) as "Aantal VHE"
from joined
where gemeten_huurstijging >= 0.0
  and gemeten_huurstijging <= 0.05
group by "Corporatie"
