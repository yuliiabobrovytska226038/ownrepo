{#
    Beleidswaarde prepared source: adds fixed Waarderingstype = 'Beleidswaarde'.
    Indexatiegebied already exists from int_chart_ontwikkeling_woningen_egw_mgw.
#}

select
    *,
    'Beleidswaarde' as "Waarderingstype"
from {{ ref("chart_vhg__beleidswaarde_source") }}
