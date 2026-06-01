{#
  Ratio of beleidswaarde/marktwaarde per COROP, excluding specific corporaties.
  Pre-computes the median ratio per COROP code for the map chart.
  Aggregates at VHE (unit) level per COROP — NOT per corporation.
  Includes Classificatie dimension (DAEB / Niet-DAEB / Totaal).
#}

with basis as (
    select
        "COROP-plusgebieden Code",
        "Classificatie",
        "Beleidswaarde",
        "Marktwaarde"
    from {{ ref("chart_waardeontwikkeling_won__basis_source") }}
    where "Beleidswaarde" is not null
      and "Marktwaarde" is not null
      and "Corporatie" not in ('Woonbedrijf SWS.Hhvl', 'Wooninc.')
),

per_classificatie as (
    select
        "Classificatie",
        "COROP-plusgebieden Code",
        median("Beleidswaarde") as median_beleidswaarde,
        median("Marktwaarde") as median_marktwaarde,
        median("Beleidswaarde") / nullif(median("Marktwaarde"), 0) as median_ratio
    from basis
    group by "Classificatie", "COROP-plusgebieden Code"
),

totaal as (
    select
        'Totaal' as "Classificatie",
        "COROP-plusgebieden Code",
        median("Beleidswaarde") as median_beleidswaarde,
        median("Marktwaarde") as median_marktwaarde,
        median("Beleidswaarde") / nullif(median("Marktwaarde"), 0) as median_ratio
    from basis
    group by "COROP-plusgebieden Code"
)

select * from per_classificatie
union all
select * from totaal
