"""Chart constants for Marktpresentatie chart assets."""

from ..config import CHART_YEAR, CHART_YEAR_M1  # noqa: F401 - re-exported for chart assets.

WAARDERINGSMODEL_ORDER = ["Woningen", "Parkeren", "BOG/MOG/ZOG"]
HANDBOEKTYPE_ORDER = [
    "EGW",
    "MGW",
    "Studenteneenheid",
    "Extramurale zorg",
    "Parkeerplaats",
    "Garagebox",
    "BOG",
    "MOG",
    "Intramurale zorg",
]
BELEIDSWAARDE_WATERFALL_COLUMNS = [
    "Marktwaarde",
    "Beschikbaarheid (scenario)",
    "Beschikbaarheid (eindwaarde)",
    "Beschikbaarheid (overdrachtskosten)",
    "Beschikbaarheid (60 jaar)",
    "Betaalbaarheid",
    "Kwaliteit (onderhoud)",
    "Kwaliteit (EFG-labels)",
    "Beheer",
    "Disconteringsvoet",
    "Beleidswaarde",
]
MARKTWAARDE_VRIJHEIDSGRADEN_GEBRUIKT = [
    "Bron disconteringsvoet (DV)",
    "Erfpacht (EP)",
    "Bron exit yield (EY)",
    "Bron leegwaarde (LW)",
    "Bron markthuur (MH)",
    "Bron Onderhoud (OH)",
    "Bron mutatieonderhoud (MOH)",
    "Bron markthuurstijging (MHS)",
    "Bron leegwaardestijging verleden (LSV)",
    "Bron leegwaardestijging (LS)",
    "Bron mutatiegraad doorexploiteren (MD)",
    "Bron mutatiegraad uitponden (MU)",
    "Bron overige kosten (OVK)",
    "Bron overige opbrengsten (OVO)",
    "Bron scenario (SC)",
    "Bron technische splitsingskosten (TS)",
    "Bron gedeelte niet verkopen bij mutatie (VM)",
]
MARKTWAARDE_VRIJHEIDSGRADEN = {
    "dv_de": {
        "name": "Disconteringsvoet doorexploiteren",
        "use_vhg": "Bron disconteringsvoet (DV)",
        "scenario": "Doorexploiteren",
        "value_vhg": "DV doorexploiteren",
        "value_basic": "DV doorexploiteren handboek",
        "value_diff": "Absoluut",
    },
    "dv_up": {
        "name": "Disconteringsvoet uitponden",
        "use_vhg": "Bron disconteringsvoet (DV)",
        "scenario": "Uitponden",
        "value_vhg": "DV uitponden",
        "value_basic": "DV uitponden handboek",
        "value_diff": "Absoluut",
    },
    "lw": {
        "name": "Leegwaarde",
        "use_vhg": "Bron leegwaarde (LW)",
        "scenario": "Uitponden",
        "value_vhg": "LW waarde",
        "value_basic": "LW handboek waarde",
        "value_diff": "Relatief",
    },
    "mh": {
        "name": "Markthuur",
        "use_vhg": "Bron markthuur (MH)",
        "scenario": "Doorexploiteren",
        "value_vhg": "MH waarde",
        "value_basic": "MH handboek waarde",
        "value_diff": "Relatief",
    },
    "oh_de": {
        "name": "Onderhoud doorexploiteren",
        "use_vhg": "Bron Onderhoud (OH)",
        "scenario": "Doorexploiteren",
        "value_vhg": "OH doorexploiteren",
        "value_basic": "OH doorexploiteren handboek",
        "value_diff": "Relatief",
    },
    "oh_up": {
        "name": "Onderhoud uitponden",
        "use_vhg": "Bron Onderhoud (OH)",
        "scenario": "Uitponden",
        "value_vhg": "OH uitponden",
        "value_basic": "OH uitponden handboek",
        "value_diff": "Relatief",
    },
    "mu_de": {
        "name": "Mutatiegraad doorexploiteren",
        "use_vhg": "Bron mutatiegraad doorexploiteren (MD)",
        "scenario": "Doorexploiteren",
        "value_vhg": "MD waarde",
        "value_basic": "MD handboek waarde",
        "value_diff": "Absoluut",
    },
    "mu_up": {
        "name": "Mutatiegraad uitponden",
        "use_vhg": "Bron mutatiegraad uitponden (MU)",
        "scenario": "Uitponden",
        "value_vhg": "MU jaar 1-15",
        "value_basic": "MU jaar 1-15 handboek",
        "value_diff": "Absoluut",
    },
    "ey_de": {
        "name": "Exit yield doorexploiteren",
        "use_vhg": "Bron exit yield (EY)",
        "scenario": "Doorexploiteren",
        "value_vhg": "EY doorexploiteren",
        "value_basic": "EY doorexploiteren handboek",
        "value_diff": "Absoluut",
    },
    "ey_up": {
        "name": "Exit yield uitponden",
        "use_vhg": "Bron exit yield (EY)",
        "scenario": "Uitponden",
        "value_vhg": "EY uitponden",
        "value_basic": "EY uitponden handboek",
        "value_diff": "Absoluut",
    },
}
BELEIDSWAARDE_VRIJHEIDSGRADEN = {
    "sh": {"name": "Beleidshuur", "value_vhg": "Beleidshuur", "value_diff": "Relatief"},
    "bb": {"name": "Beleidsbeheer", "value_vhg": "Beleidsbeheer", "value_diff": "Relatief"},
    "bo": {"name": "Beleidsonderhoud", "value_vhg": "Beleidsonderhoud", "value_diff": "Relatief"},
    "dv": {"name": "Disconteringsvoet doorexploiteren", "value_vhg": "DV doorexploiteren", "value_diff": "Relatief"},
}


def _column_values(vrijheidsgraden: dict, key: str) -> list[str]:
    return [value[key] for value in vrijheidsgraden.values() if key in value]


def _with_prior_year(columns: list[str]) -> list[str]:
    return columns + [f"{column}_{CHART_YEAR_M1}" for column in columns]


MARKTWAARDE_SCENARIO = [
    "Bron scenario (SC)",
    f"Bron scenario (SC)_{CHART_YEAR_M1}",
    "SC waarde",
    f"SC waarde_{CHART_YEAR_M1}",
]
MARKTWAARDE_SELECT = list(
    set(
        _with_prior_year(_column_values(MARKTWAARDE_VRIJHEIDSGRADEN, "use_vhg"))
        + _with_prior_year(_column_values(MARKTWAARDE_VRIJHEIDSGRADEN, "value_vhg"))
        + _with_prior_year(_column_values(MARKTWAARDE_VRIJHEIDSGRADEN, "value_basic"))
        + MARKTWAARDE_VRIJHEIDSGRADEN_GEBRUIKT
        + MARKTWAARDE_SCENARIO
    )
)
BELEIDSWAARDE_SELECT = list(set(_with_prior_year(_column_values(BELEIDSWAARDE_VRIJHEIDSGRADEN, "value_vhg"))))
