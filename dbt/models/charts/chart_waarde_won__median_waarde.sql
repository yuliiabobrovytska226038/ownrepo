with input_rows as (
    select
        corporatie,
        classificatie,
        beleidswaarde,
        marktwaarde
    from {{ ref("int_chart_woningen_egw_mgw") }}
    where corporatie not in (
        'De Woningraat',
        'Huis & Hof',
        'Woningbouwvereniging Gelderland',
        'SSH',
        'Idealis'
    )
),

base_rows as (
    select
        input_rows.corporatie,
        input_rows.classificatie,
        value_rows.waarde,
        value_rows.bedrag
    from input_rows
    cross join lateral (
        values
            ('Beleidswaarde', input_rows.beleidswaarde),
            ('Marktwaarde', input_rows.marktwaarde)
    ) as value_rows(waarde, bedrag)
    where value_rows.bedrag is not null
)

select
    waarde as "Waarde",
    case when grouping(classificatie) = 1 then 'Totaal' else classificatie end as "Classificatie",
    corporatie as "Corporatie",
    median(bedrag) as "Median waarde"
from base_rows
group by grouping sets ((waarde, corporatie, classificatie), (waarde, corporatie))
order by waarde, classificatie, corporatie
