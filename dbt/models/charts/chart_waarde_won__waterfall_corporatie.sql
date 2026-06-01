with averages as (
    select
        corporatie,
        avg(marktwaarde) as marktwaarde,
        avg("Beschikbaarheid (scenario)") as beschikbaarheid_scenario,
        avg("Beschikbaarheid (eindwaarde)") as beschikbaarheid_eindwaarde,
        avg("Beschikbaarheid (overdrachtskosten)") as beschikbaarheid_overdrachtskosten,
        avg("Beschikbaarheid (60 jaar)") as beschikbaarheid_60jaar,
        avg(betaalbaarheid) as betaalbaarheid,
        avg("Kwaliteit (onderhoud)") as kwaliteit_onderhoud,
        avg("Kwaliteit (EFG-labels)") as kwaliteit_efg,
        avg(beheer) as beheer,
        avg(disconteringsvoet) as disconteringsvoet,
        avg(beleidswaarde) as beleidswaarde
    from {{ ref("int_chart_woningen_egw_mgw") }}
    where beleidswaarde is not null
      {% if var("jaar") == 2025 %}
      and lower(trim(corporatie)) not in ('vgz (habion)')
      {% endif %}
    group by corporatie
)

select
    corporatie as "Corporatie",
    marktwaarde as "Marktwaarde",
    beschikbaarheid_scenario / nullif(marktwaarde, 0) as "Beschikbaarheid (scenario)",
    beschikbaarheid_eindwaarde / nullif(marktwaarde, 0) as "Beschikbaarheid (eindwaarde)",
    beschikbaarheid_overdrachtskosten / nullif(marktwaarde, 0) as "Beschikbaarheid (overdrachtskosten)",
    beschikbaarheid_60jaar / nullif(marktwaarde, 0) as "Beschikbaarheid (60 jaar)",
    betaalbaarheid / nullif(marktwaarde, 0) as "Betaalbaarheid",
    kwaliteit_onderhoud / nullif(marktwaarde, 0) as "Kwaliteit (onderhoud)",
    kwaliteit_efg / nullif(marktwaarde, 0) as "Kwaliteit (EFG-labels)",
    beheer / nullif(marktwaarde, 0) as "Beheer",
    disconteringsvoet / nullif(marktwaarde, 0) as "Disconteringsvoet",
    beleidswaarde as "Beleidswaarde"
from averages
