with source as (
    select
        datum,
        jaar,
        percentage,
        discount_rate,
        row_number() over (partition by jaar order by percentage) as rate_order
    from {{ source("external", "historical_data") }}
    where sheet = 'Disconteringsvoet'
)

select
    datum as "Datum",
    jaar as "Jaar",
    case
        when discount_rate is not null then cast(discount_rate as varchar)
        when rate_order = 1 then 'Doorexploiteren'
        else 'Uitponden'
    end as "Disconteringsvoet",
    percentage as "Percentage"
from source
order by
    case
        when rate_order = 1 then 1
        else 2
    end,
    jaar desc
