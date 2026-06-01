{#
    Marktwaarde prepared source: adds VTW→Eigen invoer mapping on
    Onderhoud, recomputes Waarderingstype (Full/Basis), and exposes
    the adjusted "Bron Onderhoud (OH)" column for downstream VHG models.
    Sentinel rate filtering is per-VHG and lives in the chart_vhg_filter macro.
#}

with source as (
    select
        *,
        case
            when "Bron Onderhoud (OH)" = 'VTW' then 'Eigen invoer'
            else "Bron Onderhoud (OH)"
        end as "Bron Onderhoud (OH)_adj"
    from {{ ref("chart_vhg__marktwaarde_source") }}
)

select
    * exclude ("Bron Onderhoud (OH)", "Waarderingstype", "Bron Onderhoud (OH)_adj"),
    "Bron Onderhoud (OH)_adj" as "Bron Onderhoud (OH)",
    case
        when (
            ("Bron disconteringsvoet (DV)" = 'Eigen invoer')
            or ("Erfpacht (EP)" = 'Eigen invoer')
            or ("Bron exit yield (EY)" = 'Eigen invoer')
            or ("Bron leegwaarde (LW)" = 'Eigen invoer')
            or ("Bron markthuur (MH)" = 'Eigen invoer')
            or ("Bron Onderhoud (OH)_adj" = 'Eigen invoer')
            or ("Bron mutatieonderhoud (MOH)" = 'Eigen invoer')
            or ("Bron markthuurstijging (MHS)" = 'Eigen invoer')
            or ("Bron leegwaardestijging verleden (LSV)" = 'Eigen invoer')
            or ("Bron leegwaardestijging (LS)" = 'Eigen invoer')
            or ("Bron mutatiegraad doorexploiteren (MD)" = 'Eigen invoer')
            or ("Bron mutatiegraad uitponden (MU)" = 'Eigen invoer')
            or ("Bron overige kosten (OVK)" = 'Eigen invoer')
            or ("Bron overige opbrengsten (OVO)" = 'Eigen invoer')
            or ("Bron scenario (SC)" = 'Eigen invoer')
            or ("Bron technische splitsingskosten (TS)" = 'Eigen invoer')
            or ("Bron gedeelte niet verkopen bij mutatie (VM)" = 'Eigen invoer')
        ) then 'Full'
        else 'Basis'
    end as "Waarderingstype"
from source
