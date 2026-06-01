{% macro dataset_validatie_filter_ctes() -%}
base_data as (
    select * exclude (Waarderingsmodel)
    from {{ ref('dataset_basis') }}
    where Jaar = {{ var('jaar') }}
      and Waarderingsmodel = 'Woningen'
),
corporatie_full_perc as (
    select Corporatie, avg("% Full") as "Corporatie % Full"
    from base_data
    group by Corporatie
),
step_1_full_perc as (
    select b.*
    from base_data b
    join corporatie_full_perc c on b.Corporatie = c.Corporatie
    where c."Corporatie % Full" > 0.9
),
student_zorg as (
    select distinct Corporatie, Waarderingscomplex
    from step_1_full_perc
    where Handboektype in ('Studenteneenheid', 'Extramurale zorg')
),
step_2_gemengd_complex as (
    select t.*
    from step_1_full_perc t
    anti join student_zorg sz
        on t.Corporatie = sz.Corporatie
        and t.Waarderingscomplex = sz.Waarderingscomplex
),
aardbeving_krimp as (
    select *
    from step_2_gemengd_complex
    where "Krimp/aardbeving" in ('Aardbeving', 'Beiden', 'Full')
),
step_3_aardbeving_krimp as (
    select t.*
    from step_2_gemengd_complex t
    anti join aardbeving_krimp ak
        on t.Corporatie = ak.Corporatie
        and t.Waarderingscomplex = ak.Waarderingscomplex
),
step_4_erfpacht as (
    select *
    from step_3_aardbeving_krimp
    where not (
        "Erfpacht (EP)" = 'Eigen invoer'
        and (
            "EP suppletie bij verkoop" > 0
            or "EP doorexploiteren jaar 1-15" > 0
            or "EP uitponden jaar 1-15" > 0
        )
    )
),
corporatie_aantallen as (
    select Corporatie, count(*) as Aantal
    from step_4_erfpacht
    group by Corporatie
    having count(*) > 250
),
step_5_min_250_vhes as (
    select t.*
    from step_4_erfpacht t
    join corporatie_aantallen ca
        on t.Corporatie = ca.Corporatie
)
{%- endmacro %}
