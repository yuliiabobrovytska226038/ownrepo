{% set prior_year = var("jaar") - 1 %}

with basis as (
    select
        corporatie,
        "VHE-nr",
        "COROP-plusgebieden Code",
        "COROP-plusgebieden Naam",
        "Classificatie",
        marktwaarde,
        "Marktwaarde_{{ prior_year }}" as marktwaarde_prior,
        beleidswaarde,
        "Beleidswaarde_{{ prior_year }}" as beleidswaarde_prior
    from {{ ref("int_chart_ontwikkeling_woningen_matched") }}
    where corporatie not in ('WonenBreburg')
),

expanded as (
    select
        value_type as "Waarde",
        "COROP-plusgebieden Code",
        "COROP-plusgebieden Naam",
        "Classificatie",
        marktwaarde_current,
        marktwaarde_previous,
        beleidswaarde_current,
        beleidswaarde_previous,
        value_current,
        value_previous,
        "VHE-nr"
    from basis
    cross join lateral (
        values
            ('Marktwaarde', marktwaarde, marktwaarde_prior, null::double, null::double, marktwaarde, marktwaarde_prior),
            ('Beleidswaarde', null::double, null::double, beleidswaarde, beleidswaarde_prior, beleidswaarde, beleidswaarde_prior)
    ) as value_types(value_type, marktwaarde_current, marktwaarde_previous, beleidswaarde_current, beleidswaarde_previous, value_current, value_previous)
    where value_current is not null
      and value_previous is not null
),

per_classificatie as (
    select
        "Waarde",
        "Classificatie",
        "COROP-plusgebieden Code",
        "COROP-plusgebieden Naam",
        avg(marktwaarde_current) as "Marktwaarde",
        avg(marktwaarde_previous) as "Marktwaarde_{{ prior_year }}",
        avg(beleidswaarde_current) as "Beleidswaarde",
        avg(beleidswaarde_previous) as "Beleidswaarde_{{ prior_year }}",
        avg(value_current) as value_current,
        avg(value_previous) as value_previous,
        count("VHE-nr") as "VHE-nr"
    from expanded
    group by "Waarde", "Classificatie", "COROP-plusgebieden Code", "COROP-plusgebieden Naam"
),

totaal as (
    select
        "Waarde",
        'Totaal' as "Classificatie",
        "COROP-plusgebieden Code",
        "COROP-plusgebieden Naam",
        avg(marktwaarde_current) as "Marktwaarde",
        avg(marktwaarde_previous) as "Marktwaarde_{{ prior_year }}",
        avg(beleidswaarde_current) as "Beleidswaarde",
        avg(beleidswaarde_previous) as "Beleidswaarde_{{ prior_year }}",
        avg(value_current) as value_current,
        avg(value_previous) as value_previous,
        count("VHE-nr") as "VHE-nr"
    from expanded
    group by "Waarde", "COROP-plusgebieden Code", "COROP-plusgebieden Naam"
),

grouped as (
    select * from per_classificatie
    union all
    select * from totaal
),

mutaties as (
    select
        *,
        {{ mutation_columns("value_current", "value_previous") }}
    from grouped
)

select
    "Waarde",
    "Classificatie",
    "COROP-plusgebieden Code",
    "COROP-plusgebieden Naam",
    "Marktwaarde",
    "Marktwaarde_{{ prior_year }}",
    "Beleidswaarde",
    "Beleidswaarde_{{ prior_year }}",
    "VHE-nr",
    "€ Mutatie",
    "% Mutatie",
    {{ index_mutatie('"% Mutatie"') }} as "Index Mutatie",
    {{ format_percent_text('"% Mutatie"') }} as "Text"
from mutaties
order by "Waarde", "Classificatie", "COROP-plusgebieden Code", "COROP-plusgebieden Naam"
