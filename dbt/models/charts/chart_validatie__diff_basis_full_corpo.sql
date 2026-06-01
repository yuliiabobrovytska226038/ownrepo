with diff_basis_full as (
    select
        corporatie as "Corporatie",
        sum(marktwaarde) as "Marktwaarde",
        sum({{ corrected_marktwaarde_basis() }}) as "Marktwaarde basis corr",
        count("VHE-nr") as "Aantal woningen"
    from {{ ref("dataset_validatie") }}
    group by corporatie
),

classified as (
    select
        "Corporatie",
        "Marktwaarde",
        "Marktwaarde basis corr",
        "Aantal woningen",
        "Marktwaarde basis corr" - "Marktwaarde" as "Basis min Full",
        ("Marktwaarde basis corr" - "Marktwaarde") / nullif("Marktwaarde", 0) as "Basis Full Perc"
    from diff_basis_full
)

select
    "Corporatie",
    "Marktwaarde",
    "Marktwaarde basis corr",
    "Aantal woningen",
    "Basis min Full",
    "Basis Full Perc",
    {{ basis_full_bucket('"Basis Full Perc"') }} as "Afwijking"
from classified
order by "Basis Full Perc"
