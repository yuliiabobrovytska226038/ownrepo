{% set prior_year = var("jaar") - 1 %}

with grouped as (
    select
        corporatie as "Corporatie",
        sum(beleidswaarde) as "Beleidswaarde",
        sum("Beleidswaarde_{{ prior_year }}") as "Beleidswaarde_{{ prior_year }}"
    from {{ ref("int_chart_ontwikkeling_woningen_matched") }}
    where beleidswaarde is not null
      and "Beleidswaarde_{{ prior_year }}" is not null
      and corporatie not in ('VGZ (Habion)', 'WonenBreburg', 'Huis & Hof', 'Woningbouwvereniging Gelderland')
    group by corporatie
),

mutaties as (
    select
        *,
        {{ mutation_columns('"Beleidswaarde"', '"Beleidswaarde_' ~ prior_year ~ '"') }}
    from grouped
)

select
    "Corporatie",
    "Beleidswaarde",
    "Beleidswaarde_{{ prior_year }}",
    "€ Mutatie",
    "% Mutatie",
    {{ index_mutatie('"% Mutatie"') }} as "Index Mutatie",
    {{ format_percent_text('"% Mutatie"') }} as "Text"
from mutaties
order by "Corporatie"
