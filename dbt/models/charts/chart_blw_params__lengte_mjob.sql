{% set chart_jaar = var("jaar") %}
{% set max_jaar = 60 %}

with basis as (
    select
        "VHE-nr",
        "Corporatie",
        {% for i in range(1, max_jaar + 1) %}
        case when TRY_CAST("Beleidsonderhoud jaar {{ i }}" AS DOUBLE) > 0 then 1 else 0 end
        {%- if not loop.last %} + {% endif %}
        {% endfor %}
        as "Lengte MJOB"
    from {{ source("tms", "policy_value_parameters") }}
    where "Jaar" = {{ chart_jaar }}
),

binned as (
    select
        case
            when "Lengte MJOB" = 0 then '0 (geen)'
            when "Lengte MJOB" between 1 and 15 then '1-15'
            when "Lengte MJOB" between 16 and 30 then '16-30'
            when "Lengte MJOB" between 31 and 45 then '31-45'
            when "Lengte MJOB" between 46 and 59 then '46-59'
            when "Lengte MJOB" = 60 then '60 (volledig)'
        end as "Lengte MJOB groep",
        case
            when "Lengte MJOB" = 0 then 1
            when "Lengte MJOB" between 1 and 15 then 2
            when "Lengte MJOB" between 16 and 30 then 3
            when "Lengte MJOB" between 31 and 45 then 4
            when "Lengte MJOB" between 46 and 59 then 5
            when "Lengte MJOB" = 60 then 6
        end as sort_order
    from basis
)

select
    "Lengte MJOB groep",
    sort_order,
    count(*) as "Aantal VHE"
from binned
group by "Lengte MJOB groep", sort_order
order by sort_order
