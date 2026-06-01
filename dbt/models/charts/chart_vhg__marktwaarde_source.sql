select *
from {{ ref("int_chart_ontwikkeling_woningen_egw_mgw") }}
where marktwaarde is not null
