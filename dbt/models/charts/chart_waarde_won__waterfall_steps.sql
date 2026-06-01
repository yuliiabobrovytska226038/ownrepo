{#
  Convert waterfall_corporatie from wide (one column per step) to long format.
  Applies per-step outlier exclusions that were previously hard-coded in Python.
#}

with wide as (
    select * from {{ ref("chart_waarde_won__waterfall_corporatie") }}
),

long_format as (
    select
        "Corporatie",
        step_name as "Stap",
        step_value as "Value",
        step_order as "Volgorde"
    from wide
    cross join lateral (
        values
            ('Marktwaarde', "Marktwaarde", 1),
            ('Beschikbaarheid (scenario)', "Beschikbaarheid (scenario)", 2),
            ('Beschikbaarheid (eindwaarde)', "Beschikbaarheid (eindwaarde)", 3),
            ('Beschikbaarheid (overdrachtskosten)', "Beschikbaarheid (overdrachtskosten)", 4),
            ('Beschikbaarheid (60 jaar)', "Beschikbaarheid (60 jaar)", 5),
            ('Betaalbaarheid', "Betaalbaarheid", 6),
            ('Kwaliteit (onderhoud)', "Kwaliteit (onderhoud)", 7),
            ('Kwaliteit (EFG-labels)', "Kwaliteit (EFG-labels)", 8),
            ('Beheer', "Beheer", 9),
            ('Disconteringsvoet', "Disconteringsvoet", 10),
            ('Beleidswaarde', "Beleidswaarde", 11)
    ) as steps(step_name, step_value, step_order)
)

select
    "Corporatie",
    "Stap",
    "Value",
    "Volgorde"
from long_format
where not (
    lower("Stap") = 'beschikbaarheid (scenario)'
    and lower(trim("Corporatie")) in (
        'goud wonen', 'woningstichting hulst', 'woonwaard',
        'de goede woning rijssen', 'wbo wonen', 'woningstichting tubbergen',
        'viverion', 'woongoed zeeuws vlaanderen', 'groninger huis',
        'ons huis enschede', 'domijn', 'mijande wonen',
        'kennemer wonen', 'wierden en borgen'
    )
)
and not (
    lower("Stap") = 'beschikbaarheid (overdrachtskosten)'
    and lower(trim("Corporatie")) in ('goud wonen', 'woningstichting hulst', 'sshn')
)
and not (
    lower("Stap") = 'betaalbaarheid'
    and lower(trim("Corporatie")) in ('woningbouwvereniging gelderland', 'wooninc.')
)
and not (
    lower("Stap") = 'beheer'
    and lower(trim("Corporatie")) in ('samenwerking slikkerveer', 'woonservice drenthe')
)
and not (
    lower("Stap") = 'disconteringsvoet'
    and lower(trim("Corporatie")) in ('woongoed zeeuws vlaanderen')
)
and not (
    lower("Stap") = 'marktwaarde'
    and lower(trim("Corporatie")) in ('woningbouwvereniging gelderland', 'rosehage')
)
and not (
    lower("Stap") = 'beleidswaarde'
    and lower(trim("Corporatie")) in ('woningbouwvereniging gelderland')
)
order by "Volgorde", "Corporatie"
