-- Schema: accepted_values check for Waarderingsmodel in dataset_basis.
SELECT "VHE-nr", "Corporatie", "Waarderingsmodel"
FROM {{ ref('dataset_basis') }}
WHERE "Waarderingsmodel" NOT IN (
    'Woningen', 'BOG', 'MOG', 'ZOG', 'Parkeren', 'Intramuraal',
    'BOG/MOG/ZOG', 'BOG/MOG/ZOG (harde huren)', 'Benadering'
)
