select
    avg(beleidswaarde) / nullif(avg(marktwaarde), 0) as ratio,
    corporatie as "Corporatie"
from {{ ref("int_chart_woningen_egw_mgw") }}
where beleidswaarde is not null
  and marktwaarde is not null
  and corporatie not in ('Woonbedrijf SWS.Hhvl', 'Wooninc.', 'Huis & Hof')
group by corporatie
