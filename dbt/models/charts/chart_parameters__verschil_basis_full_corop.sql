{% set chart_jaar = var("jaar") %}

with basis as (
    select
        "COROP-plusgebieden Code",
        "Classificatie",
        "Marktwaarde basis",
        "Marktwaarde_1"
    from {{ ref("dataset_basis") }}
    where "Jaar" = {{ chart_jaar }}
      and "Waarderingsmodel" = 'Woningen'
      and "Marktwaarde basis" is not null
      and "Marktwaarde_1" is not null
),

per_classificatie as (
    select
        "Classificatie",
        "COROP-plusgebieden Code",
        (avg("Marktwaarde basis") - avg("Marktwaarde_1")) / nullif(avg("Marktwaarde_1"), 0) as "Verschil basis-full"
    from basis
    group by "Classificatie", "COROP-plusgebieden Code"
),

totaal as (
    select
        'Totaal' as "Classificatie",
        "COROP-plusgebieden Code",
        (avg("Marktwaarde basis") - avg("Marktwaarde_1")) / nullif(avg("Marktwaarde_1"), 0) as "Verschil basis-full"
    from basis
    group by "COROP-plusgebieden Code"
)

select * from per_classificatie
union all
select * from totaal
