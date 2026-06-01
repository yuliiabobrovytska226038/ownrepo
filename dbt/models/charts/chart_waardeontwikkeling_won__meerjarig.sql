{#
  Multi-year value development with cumulative index.
  Combines historical (year < CHART_YEAR) + current-year t0 (all waarderingstypes + hist + CBS).
  Uses EXP(SUM(LN(...))) for cumulative product of (1 + % Mutatie).
  Replaces Python _waardeontwikkeling_won_meerjarig() and calc_waardeontwikkeling_t0(..., "Both").
#}
{% set chart_jaar = var("jaar") %}

with all_years as (
    {# Historical: all years < chart_jaar, plus chart_jaar only for types NOT provided by current_t0 #}
    select "Waarderingstype", "Jaar", "% Mutatie"
    from {{ ref("chart_waardeontwikkeling_won__historical") }}
    where "Jaar" < {{ chart_jaar }}

    union all

    select "Waarderingstype", "Jaar", "% Mutatie"
    from {{ ref("chart_waardeontwikkeling_won__historical") }}
    where "Jaar" = {{ chart_jaar }}
      and "Waarderingstype" not in ('Basis', 'Full', 'Beleidswaarde', 'Koopprijsontwikkeling (CBS)')

    union all

    select "Waarderingstype", "Jaar", "% Mutatie"
    from {{ ref("chart_waardeontwikkeling_won__current_t0") }}

    union all

    select
        'Koopprijsontwikkeling (CBS)' as "Waarderingstype",
        "Jaar",
        "mutatie_yoy" as "% Mutatie"
    from {{ ref("chart_waardeontwikkeling_won__cbs") }}
    where "Jaar" = {{ chart_jaar }}
),

with_index as (
    select
        "Waarderingstype",
        "Jaar",
        "% Mutatie",
        "% Mutatie" + 1 as "Index Mutatie",
        exp(sum(ln(case when "% Mutatie" is null then 1.0 else "% Mutatie" + 1 end))
            over (partition by "Waarderingstype" order by "Jaar"
                  rows between unbounded preceding and current row)) * 100 as "Indexcijfer",
        {{ format_percent_text('"% Mutatie"') }} as "Text"
    from all_years
),

rebased as (
    select
        *,
        "Indexcijfer" / nullif(first_value("Indexcijfer") over (
            partition by "Waarderingstype"
            order by "Jaar"
            rows between unbounded preceding and unbounded following
        ), 0) * 100 as "Indexcijfer_base_first",
        "Indexcijfer" / nullif(max(case when "Jaar" = 2016 then "Indexcijfer" end) over (
            partition by "Waarderingstype"
        ), 0) * 100 as "Indexcijfer_base_2016",
        "Indexcijfer" / nullif(max(case when "Jaar" = 2018 then "Indexcijfer" end) over (
            partition by "Waarderingstype"
        ), 0) * 100 as "Indexcijfer_base_2018"
    from with_index
)

select *
from rebased
order by "Waarderingstype", "Jaar"
