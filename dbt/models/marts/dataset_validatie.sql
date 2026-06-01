-- dataset_validatie: Quality-filtered subset of dataset_basis
-- Recreates legacy dataset_validatie from duck_query.py
-- Applies 5-step filtering: %Full >90%, no student/zorg, no aardbeving/krimp,
-- no erfpacht, min 250 VHEs per corporatie

{{ config(materialized='view') }}

WITH {{ dataset_validatie_filter_ctes() }}
SELECT * FROM step_5_min_250_vhes
