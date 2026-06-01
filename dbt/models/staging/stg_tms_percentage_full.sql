-- stg_tms_percentage_full: reconstruct the legacy Percentage Full table.
-- Full valuation and fully-free valuation use separate indicator sets.

{% set full_columns = [
    "Markthuur (MH)",
    "Leegwaarde (LW)",
    "Erfpacht (EP)",
    "Extra onderhoud (EO)",
    "Nieuw huurcontract (NHC)",
    "Looptijd bij herziening (LBH)",
    "Mutatieleegstand (MUL)",
    "Huurvrije periode na mutatie (HVP)",
    "Huurverhogingsmoment (HVM)",
    "Mutatiekosten technisch (MKT)",
    "Mutatiekosten marketing (MKM)",
    "Incentives (INC)",
    "Vaste waarde (VW)",
    "Disconteringsvoet (DV)",
    "Exit yield (EY)",
    "Scenario (SC)",
    "Onderhoud (OH)",
    "Mutatieonderhoud (MOH)",
    "Leegwaardestijging verleden (LSV)",
    "Leegwaardestijging (LS)",
    "Markthuurstijging boven inflatie (MHS)",
    "Markthuurstijging (MHS)",
    "Mutatiegraad doorexploiteren (MD)",
    "Mutatiegraad uitponden (MU)",
    "Gedeelte niet verkopen bij mutatie (VM)",
    "Technische splitsingskosten (TS)",
    "Overige kosten (OVK)",
    "Overige opbrengsten (OVO)",
] %}

{% set vrije_columns = [
    "Huurstijging (HS)",
    "Huurstijging opslag (HSO)",
    "Servicekosten eigen rekening (SK)",
    "Huurstijging boven inflatie (HS)",
    "Juridische splitsingskosten (JS)",
    "Beheerkosten (BK)",
    "Belastingen en verzekeringen (BV)",
    "OZB",
    "Huurderving (HD)",
    "Verkoopkosten (VK)",
    "Aanvangsleegstand (AL)",
    "Mutatieleegstand (ML)",
] %}

with source as (
    select *
    from {{ ref('int_marktwaardeparameters') }}
    where "VHE-nr" is not null
),

counts as (
    select
        "VHE-nr",
        "Complexcode",
        "Corporatie",
        "Jaar",
        "Peildatum",
        (
            {% for column_name in full_columns %}
            case when "{{ column_name }}" in ('Eigen invoer', 'VTW') then 1 else 0 end{{ " +" if not loop.last else "" }}
            {% endfor %}
        ) as aantal_vrijheidsgraden,
        (
            {% for column_name in vrije_columns %}
            case when "{{ column_name }}" in ('Eigen invoer', 'VTW') then 1 else 0 end{{ " +" if not loop.last else "" }}
            {% endfor %}
        ) as aantal_vrije_waardering_overrules
    from source
)

select
    "VHE-nr",
    "Complexcode",
    "Corporatie",
    "Jaar",
    "Peildatum",
    aantal_vrijheidsgraden as "Aantal vrijheidsgraden toegepast",
    case when aantal_vrijheidsgraden > 0 then 1 else 0 end as "% Full",
    aantal_vrije_waardering_overrules as "Aantal vrije waardering overrules toegepast",
    case when aantal_vrije_waardering_overrules > 0 then 1 else 0 end as "% Vrije waardering",
    case
        when aantal_vrije_waardering_overrules > 0 then 'Vrije waardering'
        when aantal_vrijheidsgraden > 0 then 'Full'
        else 'Basis'
    end as "Waarderingstype"
from counts
