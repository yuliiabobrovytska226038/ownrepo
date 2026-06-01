with combined as (
    select
        "Waarderingsmodel",
        "% Mutatie",
        {{ var("jaar") }} as "Jaar",
        "Text"
    from {{ ref("chart_waardeontwikkeling__model_current") }}

    union all

    select
        "Waarderingsmodel",
        "% Mutatie",
        "Jaar",
        "Text"
    from {{ ref("chart_waardeontwikkeling__model_historical") }}
)

select
    "Waarderingsmodel",
    "% Mutatie",
    "Jaar",
    "Text"
from combined
order by
    {{ waarderingsmodel_order('"Waarderingsmodel"') }},
    "Jaar"
