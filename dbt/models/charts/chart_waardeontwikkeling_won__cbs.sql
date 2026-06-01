select
    jaar::integer as "Jaar",
    mutatie_yoy as "mutatie_yoy"
from {{ source("external", "cbs_house_price_index") }}
where mutatie_yoy is not null
qualify row_number() over (partition by jaar order by jaar) = 1
