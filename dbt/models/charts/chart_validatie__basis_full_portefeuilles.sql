with classified as (
    select
        "Corporatie" as corporatie,
        "Aantal woningen",
        "Afwijking" as afwijking
    from {{ ref("chart_validatie__diff_basis_full_corpo") }}
),

bucket_order(afwijking, bucket_order) as (
    {{ basis_full_bucket_order_values() }}
),

bucketed as (
    select
        afwijking,
        count(corporatie) as "Aantal portefeuilles",
        sum("Aantal woningen") as "Aantal woningen"
    from classified
    group by afwijking
),

complete_buckets as (
    select
        bucket_order.afwijking,
        bucket_order.bucket_order,
        coalesce(bucketed."Aantal portefeuilles", 0) as "Aantal portefeuilles",
        coalesce(bucketed."Aantal woningen", 0) as "Aantal woningen"
    from bucket_order
    left join bucketed
      on bucket_order.afwijking = bucketed.afwijking
),

shares as (
    select
        afwijking,
        bucket_order,
        "Aantal woningen",
        "Aantal portefeuilles",
        "Aantal portefeuilles"::double / nullif(sum("Aantal portefeuilles") over (), 0) as procent,
        "Aantal woningen"::double / nullif(sum("Aantal woningen") over (), 0) as naar_woning_gewogen
    from complete_buckets
)

select
    afwijking as "Afwijking",
    "Aantal woningen",
    "Aantal portefeuilles",
    procent as "Procent",
    naar_woning_gewogen as "Naar woning gewogen",
    {{ format_percent_text("naar_woning_gewogen") }} as "Naar woning gewogen text"
from shares
order by bucket_order
