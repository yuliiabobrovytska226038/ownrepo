# Data Dictionary

Complete reference of all database schemas, tables, and key business definitions.

---

## Database Schemas

| Schema | Purpose | Populated By |
|--------|---------|--------------|
| `apis` | API metadata (company list, valuation coordinates) | Dagster assets |
| `tms` | Raw TMS report tables (Dutch column names preserved) | Dagster assets |
| `external` | Reference data (postal codes, COROP regions, GeoJSON) | Dagster assets |
| `models` | dbt-transformed tables and views | dbt |

---

## Schema: `apis`

| Table | Description | Key Columns |
|-------|-------------|-------------|
| `company_list` | All TMS-registered housing corporations | company name, customer ID |
| `company_valuation_data` | Valuation round coordinates per company × year | Corporatie, Jaar, valuation round ID, subsidiary info |

---

## Schema: `tms` — Raw TMS Report Tables

### Property Data

| Table | Description | Grain |
|-------|-------------|-------|
| `vastgoedgegevens_vhe_gegevens` | Real estate data at rental unit level | VHE × Company × Year |
| `vastgoedgegevens_waarderingscomplexen` | Real estate data at valuation complex level | Complex × Company × Year |
| `vastgoedgegevens_sturingscomplexen` | Real estate data at management complex level | Complex × Company × Year |
| `vastgoedgegevens_vov_verleden` | Historical sales (verkoopopbrengst verleden) | VHE × Company × Year |

### Valuation Results

| Table | Description | Grain |
|-------|-------------|-------|
| `valuation_overview_vhe` | Taxatieoverzicht at rental unit level | VHE × Company × Year |
| `valuation_overview_complex` | Taxatieoverzicht at complex level | Complex × Company × Year |
| `market_value_basis` | Market value basis per rental unit (JSON API) | rentalUnitInternalId × Company × Year |
| `rental_unit_mapping` | Maps TMS internal IDs to VHE numbers | rentalUnitInternalId → VHE-nr |

### Policy Value

| Table | Description | Grain |
|-------|-------------|-------|
| `policy_value_report_vhe` | Beleidswaarde report at VHE level | VHE × Company × Year |
| `policy_value_report_complex` | Beleidswaarde report at complex level | Complex × Company × Year |
| `policy_value_parameters` | Beleidswaarderingsparameters | VHE × Company × Year |

### Market Value Parameters (6 sheets)

| Table | Description | Grain |
|-------|-------------|-------|
| `market_value_parameters_vhe_param_woningen_parkeren` | VHE params: housing & parking | VHE × Company × Year |
| `market_value_parameters_vhe_param_bog_mog_zog` | VHE params: commercial/care | VHE × Company × Year |
| `market_value_parameters_vhe_param_benaderingsmethode` | VHE params: valuation method | VHE × Company × Year |
| `market_value_parameters_complexparam_woningen_parkeren` | Complex params: housing & parking | Complex × Company × Year |
| `market_value_parameters_complexparam_bog_mog_zog` | Complex params: commercial/care | Complex × Company × Year |

### Parameters Overview (24 sheets)

| Table | Level | Description |
|-------|-------|-------------|
| `parameters_overview_disconteringsvoet` | VHE | Discount rates |
| `parameters_overview_leegwaarde` | VHE | Vacant possession values |
| `parameters_overview_markthuur_woningen_parkeren` | VHE | Market rent: housing/parking |
| `parameters_overview_markthuur_bog_mog_zog` | VHE | Market rent: commercial/care |
| `parameters_overview_onderhoud_woningen_parkeren` | VHE | Maintenance: housing/parking |
| `parameters_overview_onderhoud_bog_mog_zog` | VHE | Maintenance: commercial/care |
| `parameters_overview_mutatiegraad` | VHE | Tenant turnover rates |
| `parameters_overview_exit_yield` | VHE | Exit yield parameters |
| `parameters_overview_scenario` | VHE | Valuation scenario |
| `parameters_overview_erfpacht_woningen_parkeren` | VHE | Ground lease: housing/parking |
| `parameters_overview_algemene_parameters` | Complex | General parameters |
| `parameters_overview_leegwaardestijging` | Complex | Vacant value growth |
| `parameters_overview_markthuurstijging` | Complex | Market rent growth |
| `parameters_overview_huurstijging_woningen_parkeren` | Complex | Rent increases |
| `parameters_overview_leegstand_woningen_parkeren` | Complex | Vacancy rates |
| `parameters_overview_huurbeklemming` | Complex | Rent restrictions |
| `parameters_overview_maximale_huur` | Complex | Maximum rent caps |
| `parameters_overview_overige_exploitatielasten` | Complex | Other operating costs |
| `parameters_overview_overige_kosten_en_opbrengsten` | Complex | Other costs & revenues |
| `parameters_overview_splitsingskosten` | Complex | Technical split costs |
| `parameters_overview_verkoopbeperking_woningen` | Complex | Sale restrictions |
| `parameters_overview_erfpacht_bog_mog_zog` | Complex | Ground lease: commercial |
| `parameters_overview_epv` | Complex | Energy performance allowance |
| `parameters_overview_schem_vrijheidsgraden_bog` | Complex | Freedom degrees: commercial |

### Other TMS Tables

| Table | Description | Grain |
|-------|-------------|-------|
| `ratio_rapport_vhe` | Ratio report at VHE level | VHE × Company × Year |
| `ratio_rapport_complex` | Ratio report at complex level | Complex × Company × Year |
| `complex_kenmerken` | Complex characteristics (JSON API) | Complex × Company × Year |
| `complex_references` | Complex reference comparisons (JSON API) | Complex × Company × Year |
| `tms_verschillenanalyse_marketvalue` | YoY market value difference analysis | VHE × Company × Year |
| `tms_verschillenanalyse_policyvalue` | YoY policy value difference analysis | VHE × Company × Year |

---

## Schema: `external` — Reference Data

| Table | Description | Source |
|-------|-------------|--------|
| `postal_code_mapping` | Maps postal codes → municipality → COROP → province | DLL Excel |
| `cbs_corop_regions` | CBS COROP region definitions | CBS opendata |
| `geographic_boundaries` | GeoJSON municipal boundaries | CBS |
| `cbs_house_price_index` | CBS house price index (Prijsindex Bestaande Koopwoningen) | CBS opendata |
| `dvi_housing_data` | DVI housing market statistics | External Excel |
| `historical_data` | Historical reference data | External Excel |

---

## Schema: `models` — dbt Transformed Models

### Staging

| Model | Materialization | Description |
|-------|----------------|-------------|
| `stg_tms_percentage_full` | TABLE | Reconstructs the removed "% Full" indicator from VHE market value parameters |

### Intermediate

| Model | Materialization | Description |
|-------|----------------|-------------|
| `int_vastgoedgegevens` | VIEW | Joins VHE + complex property data |
| `int_marktwaarde` | VIEW | Joins valuation overview with rental unit mapping |
| `int_beleidswaarde` | VIEW | Combines policy value report components |
| `int_marktwaardeparameters` | TABLE | Unpivots market value parameter sheets into one wide table |
| `int_parameteroverzicht_vhe` | TABLE | Joins 10 VHE-level parameter overview sheets |
| `int_parameteroverzicht_complex` | TABLE | Joins 14 complex-level parameter overview sheets |
| `int_parameteroverzicht` | TABLE | Combines VHE + complex parameter overviews |
| `int_chart_woningen_egw_mgw` | TABLE | Splits housing units by EGW (single-family) / MGW (multi-family) |
| `int_chart_ontwikkeling_woningen_matched` | TABLE | YoY matched housing units for development charts |
| `int_chart_ontwikkeling_woningen_egw_mgw` | TABLE | YoY matched housing by EGW/MGW type |

### Marts

| Model | Materialization | Description |
|-------|----------------|-------------|
| `dataset_basis` | TABLE (indexed) | Central ~100-column fact table joining all sources. Index on (Corporatie, Jaar). |
| `dataset_ontwikkeling` | TABLE (indexed) | Year-over-year comparison by VHE. Index on (Corporatie). |
| `dataset_validatie` | VIEW | Quality-filtered subset: ≥90% Full, no student/zorg, no earthquake/shrink, no ground lease, min 250 VHEs |
| `dataset_validatie_aantallen` | TABLE | Filter funnel statistics showing row counts per filtering step |

### Chart Views (~40 models)

All materialized as VIEWs. These provide pre-aggregated data for Plotly chart assets.

| Prefix | Chart Category |
|--------|---------------|
| `chart_historisch__*` | Historical discount rate and parameter trend charts |
| `chart_validatie__*` | Validation subset statistics charts |
| `chart_va__*` | Valuation analysis charts |
| `chart_vhg__*` | Property development (VHG ontwikkeling) charts |
| `chart_waarde__*` | Value distribution charts (per model, pie, bar) |
| `chart_waardeontwikkeling__*` | Value development charts (multi-year line) |
| `chart_waardeontwikkeling_won__*` | Housing-specific value development charts |
| `chart_waarde_won__*` | Housing-specific value charts |

---

## Key Business Terms (Dutch → English)

| Dutch Term | English | Description |
|-----------|---------|-------------|
| VHE (Verhuureenheid) | Rental Unit | Smallest rentable property unit |
| Waarderingscomplex | Valuation Complex | Group of VHEs valued together |
| Sturingscomplexen | Management Complex | Administrative grouping of complexes |
| Corporatie | Housing Corporation | Social housing provider |
| Werkmaatschappij | Operating Subsidiary | Business unit within a corporation |
| Marktwaarde | Market Value | Fair market value of property |
| Beleidswaarde | Policy Value | Social/regulated housing value (≤ Marktwaarde) |
| Netto huur | Net Rent | Annual net rental income |
| Doorexploiteren | Hold (DCF) | Valuation scenario: continue renting |
| Uitponden | Vacant Sale | Valuation scenario: sell units as they become vacant |
| Disconteringsvoet | Discount Rate | Rate used to discount future cash flows |
| Exit yield | Exit Yield | Capitalization rate at end of DCF horizon |
| Leegwaarde | Vacant Possession Value | Value if sold empty |
| Markthuur | Market Rent | Current market rental rate |
| Mutatiegraad | Tenant Turnover Rate | Annual rate of tenant changes |
| Erfpacht | Ground Lease | Leasehold land rights |
| WOZ-waarde | Municipal Tax Value | Government-assessed property value |
| Peildatum | Reference Date | Valuation cut-off date (typically Dec 31) |
| DAEB | Regulated Sector | Social housing segment (Diensten van Algemeen Economisch Belang) |
| Niet-DAEB | Non-Regulated Sector | Commercial/market-rate housing segment |
| Handboektype | Handbook Type | Property category per valuation handbook |
| Beschikbaarheid | Availability | Policy value component for availability discount |
| Betaalbaarheid | Affordability | Policy value component for affordability discount |
| Kwaliteit | Quality | Policy value component for quality discount |
| Beheer | Management | Policy value component for management costs |
| EGW (Eengezinswoning) | Single-Family Home | Detached/semi-detached house |
| MGW (Meergezinswoning) | Multi-Family Home | Apartment/flat |
| COROP-gebied | COROP Region | CBS statistical economic region |
| Krimp/aardbeving | Shrinkage/Earthquake | Area classification for declining/seismic zones |
| Parameteroverzicht | Parameter Overview | Comprehensive parameter listing per VHE/complex |
| % Full | Percentage Full | Binary: at least one full-valuation override applied |
| % Vrije waardering | Percentage Free Valuation | Binary: at least one fully-free override applied |
| BOG | Commercial Real Estate | Bedrijfsmatig Onroerend Goed |
| MOG | Mixed-Use Real Estate | Maatschappelijk Onroerend Goed |
| ZOG | Healthcare Real Estate | Zorgvastgoed |

---

## Column Naming Conventions

- **Raw TMS tables** preserve original Dutch column names with spaces and special characters (e.g., `"VHE-nr"`, `"Netto marktwaarde"`)
- **External tables** use normalized lowercase snake_case (via `normalize_column_names()`)
- **dbt models** use the same Dutch column names as raw sources for legacy compatibility
- **API tables** use normalized lowercase names
- Columns prefixed with `Bron ` (Source) are always strings even if values look numeric
- `_1`, `_2` suffixes indicate duplicate column names from different source sheets
