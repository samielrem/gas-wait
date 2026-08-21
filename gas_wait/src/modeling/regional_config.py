"""Regional retail ↔ wholesale mappings for exp02.

Only geographies with a defensible physical/market link to an EIA daily
gasoline spot we already have. Skipped areas are documented, not forced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data.datasets import (
    GULF_COAST_GASOLINE_REGULAR_SPOT_DAILY,
    LA_RBOB_GASOLINE_REGULAR_SPOT_DAILY,
    NY_HARBOR_GASOLINE_REGULAR_SPOT_DAILY,
    RETAIL_GASOLINE_REGULAR_CA_WEEKLY,
    RETAIL_GASOLINE_REGULAR_HOUSTON_WEEKLY,
    RETAIL_GASOLINE_REGULAR_LA_WEEKLY,
    RETAIL_GASOLINE_REGULAR_NYC_WEEKLY,
    RETAIL_GASOLINE_REGULAR_PADD1B_WEEKLY,
    RETAIL_GASOLINE_REGULAR_PADD3_WEEKLY,
    EIADatasetDefinition,
)

# Same Tuesday cutoff as exp01 so regional holdout is comparable.
NATIONAL_HOLDOUT_CUTOFF = "2016-07-19"


@dataclass(frozen=True)
class RegionalExperiment:
    region_id: str
    label: str
    retail: EIADatasetDefinition
    matched_hub: EIADatasetDefinition
    mismatched_hub: EIADatasetDefinition
    mapping_rationale: str


VIABLE_REGIONS: tuple[RegionalExperiment, ...] = (
    RegionalExperiment(
        region_id="nyc_nyh",
        label="New York City → NY Harbor",
        retail=RETAIL_GASOLINE_REGULAR_NYC_WEEKLY,
        matched_hub=NY_HARBOR_GASOLINE_REGULAR_SPOT_DAILY,
        mismatched_hub=GULF_COAST_GASOLINE_REGULAR_SPOT_DAILY,
        mapping_rationale=(
            "NYC pumps are supplied from the New York Harbor barge/pipeline complex. "
            "EIA NY Harbor conventional regular is the local wholesale print. Tightest city-hub pair on the East Coast."
        ),
    ),
    RegionalExperiment(
        region_id="padd1b_nyh",
        label="Central Atlantic (PADD 1B) → NY Harbor",
        retail=RETAIL_GASOLINE_REGULAR_PADD1B_WEEKLY,
        matched_hub=NY_HARBOR_GASOLINE_REGULAR_SPOT_DAILY,
        mismatched_hub=GULF_COAST_GASOLINE_REGULAR_SPOT_DAILY,
        mapping_rationale=(
            "PADD 1B is NY, NJ, PA, DE, MD, and DC. Those states price off NY Harbor / northern Colonial, "
            "not the Gulf Coast waterborne market. Broader than NYC but still a Harbor market."
        ),
    ),
    RegionalExperiment(
        region_id="houston_gc",
        label="Houston → Gulf Coast gasoline",
        retail=RETAIL_GASOLINE_REGULAR_HOUSTON_WEEKLY,
        matched_hub=GULF_COAST_GASOLINE_REGULAR_SPOT_DAILY,
        mismatched_hub=NY_HARBOR_GASOLINE_REGULAR_SPOT_DAILY,
        mapping_rationale=(
            "Houston sits on the U.S. Gulf refining complex. EIA Gulf Coast conventional regular is the local rack/spot analogue. "
            "NY Harbor is a destination market, not the origin."
        ),
    ),
    RegionalExperiment(
        region_id="padd3_gc",
        label="Gulf Coast (PADD 3) → Gulf Coast gasoline",
        retail=RETAIL_GASOLINE_REGULAR_PADD3_WEEKLY,
        matched_hub=GULF_COAST_GASOLINE_REGULAR_SPOT_DAILY,
        mismatched_hub=NY_HARBOR_GASOLINE_REGULAR_SPOT_DAILY,
        mapping_rationale=(
            "PADD 3 is the Gulf producing/refining region (TX, LA, AR, AL, MS, NM). "
            "The Gulf Coast spot is the regional wholesale. Broader than Houston but the same supply basin."
        ),
    ),
    RegionalExperiment(
        region_id="la_rbob",
        label="Los Angeles → LA RBOB",
        retail=RETAIL_GASOLINE_REGULAR_LA_WEEKLY,
        matched_hub=LA_RBOB_GASOLINE_REGULAR_SPOT_DAILY,
        mismatched_hub=NY_HARBOR_GASOLINE_REGULAR_SPOT_DAILY,
        mapping_rationale=(
            "Los Angeles requires CARB reformulated gasoline. EIA LA RBOB regular is the matching West Coast wholesale spec. "
            "NY Harbor conventional is a different molecule and a different logistics system."
        ),
    ),
    RegionalExperiment(
        region_id="ca_rbob",
        label="California → LA RBOB",
        retail=RETAIL_GASOLINE_REGULAR_CA_WEEKLY,
        matched_hub=LA_RBOB_GASOLINE_REGULAR_SPOT_DAILY,
        mismatched_hub=NY_HARBOR_GASOLINE_REGULAR_SPOT_DAILY,
        mapping_rationale=(
            "Statewide California retail is still CARB gasoline. LA RBOB is the primary CA wholesale benchmark. "
            "Bay Area can diverge from LA, so this is a looser match than the LA city series."
        ),
    ),
)

SKIPPED_GEOGRAPHIES: tuple[dict[str, str], ...] = (
    {
        "duoarea": "R10",
        "name": "PADD 1 East Coast",
        "reason": "Mixes Harbor-priced 1A/1B with Lower Atlantic barrels that arrive on Colonial from the Gulf. Not a single wholesale market.",
    },
    {
        "duoarea": "R1X",
        "name": "PADD 1A New England",
        "reason": "NY Harbor is the right wholesale, but NYC/PADD 1B are tighter. Skipped to keep the Northeast set small.",
    },
    {
        "duoarea": "R1Z",
        "name": "PADD 1C Lower Atlantic",
        "reason": "Supplied northbound on Colonial from the Gulf, not NY Harbor. Do not map to Harbor. Gulf Coast is origin, not the local rack.",
    },
    {
        "duoarea": "YMIA",
        "name": "Miami",
        "reason": "Lower Atlantic destination with local blend/tax effects. Gulf Coast spot is only a distant origin proxy.",
    },
    {
        "duoarea": "R50",
        "name": "PADD 5 West Coast",
        "reason": "Blends California CARB with WA/OR/AZ/NV/AK/HI conventional/other specs. LA RBOB is not this average.",
    },
    {
        "duoarea": "R5XCA",
        "name": "West Coast less California",
        "reason": "Pacific Northwest / Rockies-adjacent. We have no Seattle or PNW gasoline spot.",
    },
    {
        "duoarea": "Y48SE",
        "name": "Seattle",
        "reason": "Different spec and logistics from LA RBOB. No matching daily wholesale in the repo.",
    },
    {
        "duoarea": "Y05SF",
        "name": "San Francisco",
        "reason": "CARB like LA, but a distinct local market. LA RBOB is a sibling spec, not a local print. Parked to avoid a forced match.",
    },
    {
        "duoarea": "R20",
        "name": "PADD 2 Midwest",
        "reason": "Chicago/Group 3 pricing. No Midwest gasoline spot in the repo.",
    },
    {
        "duoarea": "R40",
        "name": "PADD 4 Rocky Mountain",
        "reason": "Isolated market. No matching daily wholesale.",
    },
    {
        "duoarea": "YORD",
        "name": "Chicago",
        "reason": "Midwest pipeline market, not NY Harbor, Gulf waterborne, or LA RBOB.",
    },
)


def retail_csv_name(region: RegionalExperiment) -> str:
    return f"{region.retail.dataset_id}.csv"


def hub_csv_name(definition: EIADatasetDefinition) -> str:
    return f"{definition.dataset_id}.csv"


def regional_model_csv(region: RegionalExperiment, processed_dir: Path) -> Path:
    return processed_dir / f"regional_weekly_model_{region.region_id}.csv"
