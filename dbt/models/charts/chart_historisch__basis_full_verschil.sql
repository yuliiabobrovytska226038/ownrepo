{% set chart_jaar = var("jaar") %}
{% set prior_year = chart_jaar - 1 %}

with current_year as (
    select
        {{ chart_jaar }} as "Jaar",
        round(
            (sum(case when "Jaar" = {{ chart_jaar }} then "Marktwaarde basis" end)
             - sum(case when "Jaar" = {{ prior_year }} then "Marktwaarde basis" end))
            / nullif(sum(case when "Jaar" = {{ prior_year }} then "Marktwaarde basis" end), 0)
            - (sum(case when "Jaar" = {{ chart_jaar }} then "Marktwaarde_1" end)
               - sum(case when "Jaar" = {{ prior_year }} then "Marktwaarde_1" end))
            / nullif(sum(case when "Jaar" = {{ prior_year }} then "Marktwaarde_1" end), 0),
            3
        ) as "Verschil basis-full"
    from {{ ref("dataset_basis") }}
    where "Waarderingsmodel" = 'Woningen'
      and "Marktwaarde basis" is not null
      and "Marktwaarde_1" is not null
),

history("Jaar", "Verschil basis-full") as (
    values
        (2021, -0.096),
        (2022, -0.030),
        (2023, 0.022),
        (2024, 0.012)
)

select "Jaar", "Verschil basis-full" from history
union all
select "Jaar", "Verschil basis-full" from current_year
