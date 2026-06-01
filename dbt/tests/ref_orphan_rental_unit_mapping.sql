-- Orphan record test: VHEs in rental_unit_mapping that don't appear in vastgoedgegevens.
-- High orphan count indicates data alignment issues between TMS API sources.
{{ config(severity='warn', warn_if='>= 100') }}
SELECT
    rum."VHE-nr",
    rum.Corporatie,
    rum.Jaar
FROM {{ source('tms', 'rental_unit_mapping') }} rum
LEFT JOIN {{ source('tms', 'vastgoedgegevens_vhe_gegevens') }} v
    ON rum."VHE-nr" = v."VHE-nummer"
    AND rum.Corporatie = v.Corporatie
    AND rum.Jaar = v.Jaar
WHERE v."VHE-nummer" IS NULL
