{% set chart_jaar = var("jaar") %}
{% set prior_year = chart_jaar - 1 %}

-- Conclusion metrics for Beleidswaarde dashboard
with ontwikkeling as (
    select
        count(*) as n_vhe,
        median(TRY_CAST("Beleidswaarde" AS DOUBLE)) as median_blw,
        median(TRY_CAST("Beleidswaarde_{{ prior_year }}" AS DOUBLE)) as median_blw_prev,
        sum(TRY_CAST("Beleidswaarde" AS DOUBLE)) as total_blw,
        sum(TRY_CAST("Beleidswaarde_{{ prior_year }}" AS DOUBLE)) as total_blw_prev,
        median(TRY_CAST("Beleidsonderhoud" AS DOUBLE)) as median_bo,
        median(TRY_CAST("Beleidsonderhoud_{{ prior_year }}" AS DOUBLE)) as median_bo_prev,
        median(TRY_CAST("Beleidsbeheer_bp" AS DOUBLE)) as median_bb,
        median(TRY_CAST("Beleidsbeheer_{{ prior_year }}" AS DOUBLE)) as median_bb_prev,
        median(TRY_CAST("Beleidshuur" AS DOUBLE)) as median_bh,
        median(TRY_CAST("Beleidshuur_{{ prior_year }}" AS DOUBLE)) as median_bh_prev
    from {{ ref("dataset_ontwikkeling") }}
    where "Waarderingsmodel" = 'Woningen'
      and _merge = 'both'
),

ratio as (
    select
        median(TRY_CAST("Beleidswaarde" AS DOUBLE) / NULLIF(TRY_CAST("Marktwaarde" AS DOUBLE), 0)) as median_ratio_blw_mw
    from {{ ref("dataset_basis") }}
    where "Waarderingsmodel" = 'Woningen'
      and TRY_CAST("Marktwaarde" AS DOUBLE) > 0
),

drivers as (
    select
        "Stap",
        "Mutatie",
        "Cumulatief"
    from {{ ref("chart_va__policyvalue_drivers") }}
),

segments as (
    select
        "Segment huurregime",
        count(*) as n_segment
    from {{ source("tms", "policy_value_parameters") }}
    where "Jaar" = {{ chart_jaar }}
    group by "Segment huurregime"
)

select
    'Beleidswaarde' as metric_type,
    o.n_vhe,
    o.median_blw,
    o.median_blw_prev,
    (o.median_blw - o.median_blw_prev) / o.median_blw_prev as pct_change_median,
    o.total_blw,
    o.total_blw_prev,
    (o.total_blw - o.total_blw_prev) / o.total_blw_prev as pct_change_total,
    o.median_bo,
    o.median_bo_prev,
    (o.median_bo - o.median_bo_prev) / o.median_bo_prev as pct_change_bo,
    o.median_bb,
    o.median_bb_prev,
    (o.median_bb - o.median_bb_prev) / o.median_bb_prev as pct_change_bb,
    o.median_bh,
    o.median_bh_prev,
    (o.median_bh - o.median_bh_prev) / o.median_bh_prev as pct_change_bh,
    r.median_ratio_blw_mw,
    d."Stap" as driver_stap,
    d."Mutatie" as driver_mutatie,
    d."Cumulatief" as driver_cumulatief,
    s."Segment huurregime" as segment,
    s.n_segment
from ontwikkeling o
cross join ratio r
left join drivers d on 1=1
left join segments s on 1=1
