{% set chart_jaar = var("jaar") %}
{% set prior_year = chart_jaar - 1 %}

-- Conclusion metrics for Marktwaarde dashboard
with ontwikkeling as (
    select
        "Waarderingstype",
        count(*) as n_vhe,
        median(TRY_CAST("Marktwaarde" AS DOUBLE)) as median_mw,
        median(TRY_CAST("Marktwaarde_{{ prior_year }}" AS DOUBLE)) as median_mw_prev,
        sum(TRY_CAST("Marktwaarde" AS DOUBLE)) as total_mw,
        sum(TRY_CAST("Marktwaarde_{{ prior_year }}" AS DOUBLE)) as total_mw_prev
    from {{ ref("dataset_ontwikkeling") }}
    where "Waarderingsmodel" = 'Woningen'
      and _merge = 'both'
    group by "Waarderingstype"
),

parameters as (
    select
        "Waarderingstype",
        median(TRY_CAST("DV doorexploiteren" AS DOUBLE)) as median_dv,
        median(TRY_CAST("EY doorexploiteren" AS DOUBLE)) as median_ey,
        median(TRY_CAST("MU jaar 1-15" AS DOUBLE)) as median_mu
    from {{ ref("dataset_basis") }}
    where "Waarderingsmodel" = 'Woningen'
    group by "Waarderingstype"
),

drivers as (
    select
        "Waarderingstype",
        "Stap",
        "Mutatie",
        "Cumulatief"
    from {{ ref("chart_va__marketvalue_drivers") }}
)

select
    o."Waarderingstype",
    o.n_vhe,
    o.median_mw,
    o.median_mw_prev,
    (o.median_mw - o.median_mw_prev) / o.median_mw_prev as pct_change_median,
    o.total_mw,
    o.total_mw_prev,
    (o.total_mw - o.total_mw_prev) / o.total_mw_prev as pct_change_total,
    p.median_dv,
    p.median_ey,
    p.median_mu,
    d."Stap" as driver_stap,
    d."Mutatie" as driver_mutatie,
    d."Cumulatief" as driver_cumulatief
from ontwikkeling o
left join parameters p on o."Waarderingstype" = p."Waarderingstype"
left join drivers d on o."Waarderingstype" = d."Waarderingstype"
