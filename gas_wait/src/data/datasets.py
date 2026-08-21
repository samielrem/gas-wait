"""Verified EIA dataset definitions for the Gas Wait pipeline.

Each definition below was checked against official EIA API v2 documentation
and the EIA Open Data browser URLs for the corresponding petroleum routes.

References:
- API docs: https://www.eia.gov/opendata/documentation.php
- Retail gasoline route: https://www.eia.gov/opendata/browser/petroleum/pri/gnd
- Crude spot prices route: https://www.eia.gov/opendata/browser/petroleum/pri/spt
- Weekly stocks route: https://www.eia.gov/opendata/browser/petroleum/stoc/wstk
- Spot price table: https://www.eia.gov/dnav/pet/PET_PRI_SPT_S1_D.htm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EIADatasetDefinition:
    dataset_id: str
    name: str
    route: str
    frequency: str
    facets: dict[str, list[str]]
    geographic_area: str
    units: str
    category: str
    source: str = "U.S. Energy Information Administration (EIA) API v2"
    data_columns: list[str] = field(default_factory=lambda: ["value"])
    legacy_series_id: str | None = None
    description: str = ""

    def to_metadata(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "route": self.route,
            "frequency": self.frequency,
            "facets": self.facets,
            "geographic_area": self.geographic_area,
            "units": self.units,
            "category": self.category,
            "source": self.source,
            "legacy_series_id": self.legacy_series_id,
            "description": self.description,
        }


# Retail gasoline: U.S. regular gasoline retail price, weekly.
# EIA browser facets: product=EPMR, duoarea=NUS, process=PTE
# Legacy v1 series: PET.EMM_EPMR_PTE_NUS_DPG.W
RETAIL_GASOLINE_REGULAR_US_WEEKLY = EIADatasetDefinition(
    dataset_id="retail_gasoline_regular_us_weekly",
    name="U.S. Regular Gasoline Retail Price",
    route="petroleum/pri/gnd/data",
    frequency="weekly",
    facets={
        "product": ["EPMR"],
        "duoarea": ["NUS"],
        "process": ["PTE"],
    },
    geographic_area="United States",
    units="dollars per gallon",
    category="retail_gasoline_price",
    legacy_series_id="PET.EMM_EPMR_PTE_NUS_DPG.W",
    description=(
        "Weekly U.S. retail price for regular grade gasoline. "
        "Published as part of EIA's Gasoline and Diesel Fuel Update."
    ),
)

# Crude oil: WTI Cushing spot price, weekly.
# EIA browser facet: series=RWTC on petroleum/pri/spt
# Legacy v1 series: PET.RWTC.W
WTI_CRUDE_SPOT_US_WEEKLY = EIADatasetDefinition(
    dataset_id="wti_crude_spot_us_weekly",
    name="WTI Crude Oil Spot Price",
    route="petroleum/pri/spt/data",
    frequency="weekly",
    facets={
        "series": ["RWTC"],
    },
    geographic_area="United States (Cushing, OK)",
    units="dollars per barrel",
    category="crude_oil_price",
    legacy_series_id="PET.RWTC.W",
    description=(
        "Weekly WTI crude oil spot price. Crude costs are the largest component "
        "of retail gasoline prices."
    ),
)

# Gasoline inventories: total U.S. gasoline stocks, weekly.
# EIA browser facet: series=WGTSTUS1 on petroleum/stoc/wstk
GASOLINE_INVENTORIES_US_WEEKLY = EIADatasetDefinition(
    dataset_id="gasoline_inventories_us_weekly",
    name="U.S. Total Gasoline Stocks",
    route="petroleum/stoc/wstk/data",
    frequency="weekly",
    facets={
        "series": ["WGTSTUS1"],
    },
    geographic_area="United States",
    units="million barrels",
    category="gasoline_inventories",
    legacy_series_id="WGTSTUS1",
    description=(
        "Weekly U.S. total gasoline stocks from the Weekly Petroleum Status Report. "
        "Inventory builds can signal oversupply; draws can signal tight supply."
    ),
)

ACTIVE_WEEKLY_DATASETS: tuple[EIADatasetDefinition, ...] = (
    RETAIL_GASOLINE_REGULAR_US_WEEKLY,
    WTI_CRUDE_SPOT_US_WEEKLY,
    GASOLINE_INVENTORIES_US_WEEKLY,
)

# Daily petroleum spot prices on petroleum/pri/spt.
# Verified via EIA API v2 (frequency=daily) and legacy seriesid lookups on 2026-08-20.

# WTI Cushing spot, daily.
# EIA browser facet: series=RWTC
# Legacy v1 series: PET.RWTC.D
WTI_CRUDE_SPOT_US_DAILY = EIADatasetDefinition(
    dataset_id="wti_crude_spot_us_daily",
    name="WTI Crude Oil Spot Price",
    route="petroleum/pri/spt/data",
    frequency="daily",
    facets={
        "series": ["RWTC"],
    },
    geographic_area="United States (Cushing, OK)",
    units="dollars per barrel",
    category="crude_oil_price",
    legacy_series_id="PET.RWTC.D",
    description=(
        "Daily WTI crude oil spot price at Cushing, OK. Wholesale benchmark that "
        "feeds into gasoline costs."
    ),
)

# New York Harbor conventional regular gasoline spot, daily.
# EIA browser facet: series=EER_EPMRU_PF4_Y35NY_DPG
# Legacy v1 series: PET.EER_EPMRU_PF4_Y35NY_DPG.D
NY_HARBOR_GASOLINE_REGULAR_SPOT_DAILY = EIADatasetDefinition(
    dataset_id="ny_harbor_gasoline_regular_spot_daily",
    name="New York Harbor Conventional Gasoline Regular Spot Price",
    route="petroleum/pri/spt/data",
    frequency="daily",
    facets={
        "series": ["EER_EPMRU_PF4_Y35NY_DPG"],
    },
    geographic_area="New York Harbor",
    units="dollars per gallon",
    category="wholesale_gasoline_price",
    legacy_series_id="PET.EER_EPMRU_PF4_Y35NY_DPG.D",
    description=(
        "Daily New York Harbor conventional regular gasoline spot price FOB. "
        "East Coast wholesale benchmark."
    ),
)

# U.S. Gulf Coast conventional regular gasoline spot, daily.
# EIA browser facet: series=EER_EPMRU_PF4_RGC_DPG
# Legacy v1 series: PET.EER_EPMRU_PF4_RGC_DPG.D
GULF_COAST_GASOLINE_REGULAR_SPOT_DAILY = EIADatasetDefinition(
    dataset_id="gulf_coast_gasoline_regular_spot_daily",
    name="U.S. Gulf Coast Conventional Gasoline Regular Spot Price",
    route="petroleum/pri/spt/data",
    frequency="daily",
    facets={
        "series": ["EER_EPMRU_PF4_RGC_DPG"],
    },
    geographic_area="U.S. Gulf Coast",
    units="dollars per gallon",
    category="wholesale_gasoline_price",
    legacy_series_id="PET.EER_EPMRU_PF4_RGC_DPG.D",
    description=(
        "Daily U.S. Gulf Coast conventional regular gasoline spot price FOB. "
        "Gulf Coast wholesale benchmark."
    ),
)

# Los Angeles RBOB regular gasoline spot, daily.
# EIA browser facet: series=EER_EPMRR_PF4_Y05LA_DPG
# Legacy v1 series: PET.EER_EPMRR_PF4_Y05LA_DPG.D
LA_RBOB_GASOLINE_REGULAR_SPOT_DAILY = EIADatasetDefinition(
    dataset_id="la_rbob_gasoline_regular_spot_daily",
    name="Los Angeles RBOB Regular Gasoline Spot Price",
    route="petroleum/pri/spt/data",
    frequency="daily",
    facets={
        "series": ["EER_EPMRR_PF4_Y05LA_DPG"],
    },
    geographic_area="Los Angeles, CA",
    units="dollars per gallon",
    category="wholesale_gasoline_price",
    legacy_series_id="PET.EER_EPMRR_PF4_Y05LA_DPG.D",
    description=(
        "Daily Los Angeles reformulated RBOB regular gasoline spot price. "
        "West Coast wholesale benchmark."
    ),
)

ACTIVE_DAILY_DATASETS: tuple[EIADatasetDefinition, ...] = (
    WTI_CRUDE_SPOT_US_DAILY,
    NY_HARBOR_GASOLINE_REGULAR_SPOT_DAILY,
    GULF_COAST_GASOLINE_REGULAR_SPOT_DAILY,
    LA_RBOB_GASOLINE_REGULAR_SPOT_DAILY,
)

# Backward-compatible alias for the original weekly-only pipeline.
ACTIVE_DATASETS: tuple[EIADatasetDefinition, ...] = ACTIVE_WEEKLY_DATASETS

# Weekly regular retail by geography (Form EIA-878 / Gasoline and Diesel Fuel Update).
# Same route/product/process as the national series; only duoarea changes.
# Not included in ALL_DATASETS so the original fetch pipeline is unchanged.


def _regional_retail(
    dataset_id: str,
    name: str,
    duoarea: str,
    geographic_area: str,
    legacy_series_id: str,
) -> EIADatasetDefinition:
    return EIADatasetDefinition(
        dataset_id=dataset_id,
        name=name,
        route="petroleum/pri/gnd/data",
        frequency="weekly",
        facets={
            "product": ["EPMR"],
            "duoarea": [duoarea],
            "process": ["PTE"],
        },
        geographic_area=geographic_area,
        units="dollars per gallon",
        category="retail_gasoline_price",
        legacy_series_id=legacy_series_id,
        description=(
            f"Weekly regular-grade retail gasoline price for {geographic_area}. "
            "Published in EIA's Gasoline and Diesel Fuel Update."
        ),
    )


RETAIL_GASOLINE_REGULAR_NYC_WEEKLY = _regional_retail(
    "retail_gasoline_regular_new_york_city_weekly",
    "New York City Regular Gasoline Retail Price",
    "Y35NY",
    "New York City",
    "PET.EMM_EPMR_PTE_Y35NY_DPG.W",
)
RETAIL_GASOLINE_REGULAR_PADD1B_WEEKLY = _regional_retail(
    "retail_gasoline_regular_padd1b_weekly",
    "Central Atlantic (PADD 1B) Regular Gasoline Retail Price",
    "R1Y",
    "PADD 1B Central Atlantic",
    "PET.EMM_EPMR_PTE_R1Y_DPG.W",
)
RETAIL_GASOLINE_REGULAR_HOUSTON_WEEKLY = _regional_retail(
    "retail_gasoline_regular_houston_weekly",
    "Houston Regular Gasoline Retail Price",
    "Y44HO",
    "Houston, TX",
    "PET.EMM_EPMR_PTE_Y44HO_DPG.W",
)
RETAIL_GASOLINE_REGULAR_PADD3_WEEKLY = _regional_retail(
    "retail_gasoline_regular_padd3_weekly",
    "Gulf Coast (PADD 3) Regular Gasoline Retail Price",
    "R30",
    "PADD 3 Gulf Coast",
    "PET.EMM_EPMR_PTE_R30_DPG.W",
)
RETAIL_GASOLINE_REGULAR_LA_WEEKLY = _regional_retail(
    "retail_gasoline_regular_los_angeles_weekly",
    "Los Angeles Regular Gasoline Retail Price",
    "Y05LA",
    "Los Angeles, CA",
    "PET.EMM_EPMR_PTE_Y05LA_DPG.W",
)
RETAIL_GASOLINE_REGULAR_CA_WEEKLY = _regional_retail(
    "retail_gasoline_regular_california_weekly",
    "California Regular Gasoline Retail Price",
    "SCA",
    "California",
    "PET.EMM_EPMR_PTE_SCA_DPG.W",
)

REGIONAL_RETAIL_DATASETS: tuple[EIADatasetDefinition, ...] = (
    RETAIL_GASOLINE_REGULAR_NYC_WEEKLY,
    RETAIL_GASOLINE_REGULAR_PADD1B_WEEKLY,
    RETAIL_GASOLINE_REGULAR_HOUSTON_WEEKLY,
    RETAIL_GASOLINE_REGULAR_PADD3_WEEKLY,
    RETAIL_GASOLINE_REGULAR_LA_WEEKLY,
    RETAIL_GASOLINE_REGULAR_CA_WEEKLY,
)

ALL_DATASETS: tuple[EIADatasetDefinition, ...] = (
    *ACTIVE_WEEKLY_DATASETS,
    *ACTIVE_DAILY_DATASETS,
)
