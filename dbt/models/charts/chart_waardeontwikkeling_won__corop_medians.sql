{#
  Pre-computed COROP-plusgebied median aggregations for map charts.
  Replaces 8 separate Python groupby().agg(median) calls with one SQL model.
  Each row = one COROP code + Classificatie with all relevant median values.
  Aggregates at VHE (unit) level per COROP.
  Includes Classificatie dimension (DAEB / Niet-DAEB / Totaal).
#}

with basis as (
    select * from {{ ref("chart_waardeontwikkeling_won__basis_source") }}
),

per_classificatie as (
    select
        "Classificatie",
        "COROP-plusgebieden Code",
        median("Beleidsbeheer_bp") as median_beleidsbeheer,
        median("Beleidshuur") as median_beleidshuur,
        median("Beleidswaarde") as median_beleidswaarde,
        median("Marktwaarde") as median_marktwaarde,
        median("LW waarde") as median_LW,
        median("Beleidsonderhoud") as median_beleidsonderhoud,
        median("DV doorexploiteren") as median_DV
    from basis
    group by "Classificatie", "COROP-plusgebieden Code"
),

totaal as (
    select
        'Totaal' as "Classificatie",
        "COROP-plusgebieden Code",
        median("Beleidsbeheer_bp") as median_beleidsbeheer,
        median("Beleidshuur") as median_beleidshuur,
        median("Beleidswaarde") as median_beleidswaarde,
        median("Marktwaarde") as median_marktwaarde,
        median("LW waarde") as median_LW,
        median("Beleidsonderhoud") as median_beleidsonderhoud,
        median("DV doorexploiteren") as median_DV
    from basis
    group by "COROP-plusgebieden Code"
)

select * from per_classificatie
union all
select * from totaal
