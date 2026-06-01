with base_rows as (
    select *
    from {{ ref("int_chart_woningen_egw_mgw") }}
    where beleidswaarde is not null
),

classified as (
    select 'Totaal' as classificatie, * from base_rows
    union all
    select classificatie, * from base_rows where classificatie in ('DAEB', 'Niet-DAEB')
),

averages as (
    select
        classificatie,
        avg(marktwaarde) as avg_marktwaarde,
        avg("Beschikbaarheid (scenario)") as avg_beschikbaarheid_scenario,
        avg("Beschikbaarheid (eindwaarde)") as avg_beschikbaarheid_eindwaarde,
        avg("Beschikbaarheid (overdrachtskosten)") as avg_beschikbaarheid_overdrachtskosten,
        avg("Beschikbaarheid (60 jaar)") as avg_beschikbaarheid_60jaar,
        avg(betaalbaarheid) as avg_betaalbaarheid,
        avg("Kwaliteit (onderhoud)") as avg_kwaliteit_onderhoud,
        avg("Kwaliteit (EFG-labels)") as avg_kwaliteit_efg,
        avg(beheer) as avg_beheer,
        avg(disconteringsvoet) as avg_disconteringsvoet,
        avg(beleidswaarde) as avg_beleidswaarde
    from classified
    group by classificatie
),

unpivoted as (
    select
        classificatie,
        step.stap,
        step.waarde,
        step.volgorde
    from averages
    cross join lateral (
        values
            ('Marktwaarde', avg_marktwaarde, 1),
            ('Beschikbaarheid (scenario)', avg_beschikbaarheid_scenario, 2),
            ('Beschikbaarheid (eindwaarde)', avg_beschikbaarheid_eindwaarde, 3),
            ('Beschikbaarheid (overdrachtskosten)', avg_beschikbaarheid_overdrachtskosten, 4),
            ('Beschikbaarheid (60 jaar)', avg_beschikbaarheid_60jaar, 5),
            ('Betaalbaarheid', avg_betaalbaarheid, 6),
            ('Kwaliteit (onderhoud)', avg_kwaliteit_onderhoud, 7),
            ('Kwaliteit (EFG-labels)', avg_kwaliteit_efg, 8),
            ('Beheer', avg_beheer, 9),
            ('Disconteringsvoet', avg_disconteringsvoet, 10),
            ('Beleidswaarde', avg_beleidswaarde, 11)
    ) as step(stap, waarde, volgorde)
),

percentages as (
    select
        classificatie,
        stap,
        waarde,
        waarde / nullif(first_value(waarde) over (partition by classificatie order by volgorde), 0) as procent,
        volgorde
    from unpivoted
)

select
    classificatie as "Classificatie",
    stap as "Stap",
    waarde as "Waarde",
    procent as "Procent",
    {{ format_percent_text("procent") }} as "Text",
    volgorde as "Volgorde",
    case when volgorde = 1 then 'absolute' when volgorde = 11 then 'total' else 'relative' end as "Measure"
from percentages
order by classificatie, volgorde
