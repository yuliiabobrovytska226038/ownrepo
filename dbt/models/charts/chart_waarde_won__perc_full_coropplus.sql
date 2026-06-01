{% set chart_jaar = var("jaar") %}

with base as (
    select
        "COROP-plusgebieden Code",
        "COROP-plusgebieden Naam",
        "Classificatie",
        "% Full"
    from {{ ref("dataset_basis") }}
    where jaar = {{ chart_jaar }}
      and waarderingsmodel = 'Woningen'
),

-- Per Classificatie, per COROP
per_class_per_corop as (
    select
        "Classificatie",
        "COROP-plusgebieden Code",
        "COROP-plusgebieden Naam",
        avg("% Full") as "% Full"
    from base
    group by "Classificatie", "COROP-plusgebieden Code", "COROP-plusgebieden Naam"
),

-- Per Classificatie, Nederland totaal
per_class_nederland as (
    select
        "Classificatie",
        'Nederland' as "COROP-plusgebieden Code",
        '' as "COROP-plusgebieden Naam",
        avg("% Full") as "% Full"
    from base
    group by "Classificatie"
),

-- Totaal Classificatie, per COROP
totaal_per_corop as (
    select
        'Totaal' as "Classificatie",
        "COROP-plusgebieden Code",
        "COROP-plusgebieden Naam",
        avg("% Full") as "% Full"
    from base
    group by "COROP-plusgebieden Code", "COROP-plusgebieden Naam"
),

-- Totaal Classificatie, Nederland totaal
totaal_nederland as (
    select
        'Totaal' as "Classificatie",
        'Nederland' as "COROP-plusgebieden Code",
        '' as "COROP-plusgebieden Naam",
        avg("% Full") as "% Full"
    from base
)

select * from per_class_per_corop
union all
select * from per_class_nederland
union all
select * from totaal_per_corop
union all
select * from totaal_nederland
