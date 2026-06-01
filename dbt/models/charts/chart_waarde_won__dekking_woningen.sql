{% set chart_jaar = var("jaar") %}

with onderzoek as (
    select
        "Provincies Code",
        "Provincies Naam",
        count("VHE-nr") as "Aantal woningen onderzoek"
    from {{ ref("dataset_basis") }}
    where jaar = {{ chart_jaar }}
      and waarderingsmodel = 'Woningen'
    group by "Provincies Code", "Provincies Naam"
),

dvi as (
    select
        c.provincie_code as "Provincies Code",
        c.provincie_naam as "Provincies Naam",
        sum(d.aantal_ultimo) as "Aantal woningen totaal"
    from {{ source("external", "dvi_housing_data") }} as d
    join {{ source("external", "cbs_corop_regions") }} as c
      on 'GM' || lpad(cast(cast(d.gemeente_code as integer) as varchar), 4, '0') = c.gemeentecode
    where d.sheet = 'Gemeente aantal ultimo'
      and d.gemeente_code is not null
      and d.eenheid_soort in ('WoonZelfst', 'WoonOnzelfst')
    group by c.provincie_code, c.provincie_naam
),

joined as (
    select
        coalesce(onderzoek."Provincies Code", dvi."Provincies Code") as "Provincies Code",
        coalesce(onderzoek."Provincies Naam", dvi."Provincies Naam") as "Provincies Naam",
        onderzoek."Aantal woningen onderzoek",
        dvi."Aantal woningen totaal",
        onderzoek."Aantal woningen onderzoek" / nullif(dvi."Aantal woningen totaal", 0) as "Dekking naar woningen"
    from onderzoek
    full outer join dvi
      on onderzoek."Provincies Code" = dvi."Provincies Code"
     and onderzoek."Provincies Naam" = dvi."Provincies Naam"
)

select
    *
from joined
where "Provincies Code" is not null
  and "Provincies Code" != '-'
