-- Orphan record test: VHEs in valuation_overview_vhe without matching property data in vastgoedgegevens.
-- High orphan count indicates data alignment issues between TMS API sources.
{{ config(severity='warn', warn_if='>= 100') }}
SELECT
    w."VHE-nr",
    w.Corporatie,
    w.Jaar
FROM {{ source('tms', 'valuation_overview_vhe') }} w
LEFT JOIN {{ source('tms', 'vastgoedgegevens_vhe_gegevens') }} v
    ON w."VHE-nr" = v."VHE-nummer"
    AND w.Corporatie = v.Corporatie
    AND w.Jaar = v.Jaar
WHERE v."VHE-nummer" IS NULL
