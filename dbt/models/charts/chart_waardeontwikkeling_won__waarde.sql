{% set prior_year = var("jaar") - 1 %}

with basis as (
    select
        "VHE-nr",
        marktwaarde,
        "Marktwaarde_{{ prior_year }}" as marktwaarde_prior,
        beleidswaarde,
        "Beleidswaarde_{{ prior_year }}" as beleidswaarde_prior,
        waarderingstype,
        "Waarderingstype_{{ prior_year }}" as waarderingstype_prior,
        jaar
    from {{ ref("int_chart_ontwikkeling_woningen_matched") }}
),

expanded as (
    select
        value_filter as "WaardeFilter",
        value_waarderingstype as "Waarderingstype",
        jaar as "Jaar",
        value_current,
        value_prior,
        "VHE-nr"
    from basis
    cross join lateral (
        values
            ('Marktwaarde', waarderingstype, marktwaarde, marktwaarde_prior, true),
            ('Beleidswaarde', 'Beleidswaarde', beleidswaarde, beleidswaarde_prior, false),
            ('Beleidswaarde_uitgesplitst', waarderingstype, beleidswaarde, beleidswaarde_prior, true)
    ) as value_types(value_filter, value_waarderingstype, value_current, value_prior, require_same_type)
    where value_current is not null
      and value_prior is not null
      and (not require_same_type or waarderingstype = waarderingstype_prior)
),

grouped as (
    select
        "WaardeFilter",
        "Waarderingstype",
        "Jaar",
        avg(value_current) as value_current,
        avg(value_prior) as value_prior,
        count("VHE-nr") as "Aantal VHE"
    from expanded
    group by "WaardeFilter", "Waarderingstype", "Jaar"
),

mutaties as (
    select
        *,
        {{ mutation_columns("value_current", "value_prior") }}
    from grouped
)

select
    "WaardeFilter",
    "Waarderingstype",
    "Jaar",
    case when "WaardeFilter" = 'Marktwaarde' then value_current end as "Marktwaarde",
    case when "WaardeFilter" = 'Marktwaarde' then value_prior end as "Marktwaarde_{{ prior_year }}",
    case when "WaardeFilter" <> 'Marktwaarde' then value_current end as "Beleidswaarde",
    case when "WaardeFilter" <> 'Marktwaarde' then value_prior end as "Beleidswaarde_{{ prior_year }}",
    "Aantal VHE",
    "% Mutatie" as "Waardeontwikkeling",
    "€ Mutatie",
    "% Mutatie",
    {{ index_mutatie('"% Mutatie"') }} as "Index Mutatie",
    {{ format_percent_text('"% Mutatie"') }} as "Text"
from mutaties
order by "WaardeFilter", "Waarderingstype", "Jaar"
