"""
carbon_calculator.py — Economic & Carbon Impact Calculator
Converts deforested hectares into real-world CO₂ and financial metrics.

Coefficients based on IPCC tropical forest carbon stock estimates.
"""

from dataclasses import dataclass
from typing import Optional


# ─── Biome Coefficients ───────────────────────────────────────────
# Source: IPCC Guidelines for National Greenhouse Gas Inventories
# Units: tonnes CO₂ per hectare

BIOME_COEFFICIENTS = {
    "Amazon Rainforest (Dense)":     {"co2_per_ha": 300, "biodiversity": "Extremely High", "emoji": "🌿"},
    "Amazon Rainforest (Moderate)":  {"co2_per_ha": 250, "biodiversity": "Very High",      "emoji": "🌱"},
    "Cerrado (Brazilian Savanna)":   {"co2_per_ha": 150, "biodiversity": "High",           "emoji": "🌾"},
    "Atlantic Forest":               {"co2_per_ha": 280, "biodiversity": "Extremely High", "emoji": "🍃"},
    "Pantanal (Wetland)":            {"co2_per_ha": 180, "biodiversity": "High",           "emoji": "🌊"},
}

# ─── Carbon Credit Market Prices (USD per tonne CO₂) ─────────────
CARBON_MARKETS = {
    "Voluntary Market (VCM)":        {"price": 15,  "description": "Average 2024 voluntary carbon credit"},
    "California Cap-and-Trade":      {"price": 35,  "description": "California compliance market"},
    "EU Emissions Trading (ETS)":    {"price": 65,  "description": "European compliance market"},
    "Article 6 (Paris Agreement)":   {"price": 50,  "description": "International carbon trading"},
    "REDD+ Credits":                 {"price": 20,  "description": "Reducing Emissions from Deforestation"},
}

# ─── Additional ecosystem values ─────────────────────────────────
ECOSYSTEM_VALUES = {
    "water_regulation_usd_per_ha":   800,    # Annual water regulation services
    "biodiversity_usd_per_ha":       1200,   # Biodiversity habitat value
    "timber_usd_per_ha":             2500,   # Standing timber value
    "soil_carbon_usd_per_ha":        400,    # Soil carbon storage
}


@dataclass
class CarbonImpact:
    hectares: float
    biome: str
    co2_per_ha: float
    total_co2_tonnes: float
    carbon_market: str
    carbon_price_usd: float
    carbon_value_usd: float
    ecosystem_value_usd: float
    total_economic_loss_usd: float
    trees_equivalent: int
    cars_equivalent: int
    homes_equivalent: int


def calculate_carbon_impact(
    hectares: float,
    biome: str = "Amazon Rainforest (Dense)",
    carbon_market: str = "Voluntary Market (VCM)"
) -> CarbonImpact:
    """
    Calculate full carbon and economic impact of deforestation.

    Args:
        hectares      : Area of forest lost (ha)
        biome         : Forest biome type (see BIOME_COEFFICIENTS keys)
        carbon_market : Carbon credit market (see CARBON_MARKETS keys)

    Returns:
        CarbonImpact dataclass with all metrics
    """
    coeff  = BIOME_COEFFICIENTS.get(biome, BIOME_COEFFICIENTS["Amazon Rainforest (Dense)"])
    market = CARBON_MARKETS.get(carbon_market, CARBON_MARKETS["Voluntary Market (VCM)"])

    co2_per_ha       = coeff["co2_per_ha"]
    total_co2        = hectares * co2_per_ha
    carbon_price     = market["price"]
    carbon_value     = total_co2 * carbon_price

    # Ecosystem services (annual × 10-year NPV at 5% discount)
    ecosystem_annual = (
        ECOSYSTEM_VALUES["water_regulation_usd_per_ha"] +
        ECOSYSTEM_VALUES["biodiversity_usd_per_ha"] +
        ECOSYSTEM_VALUES["soil_carbon_usd_per_ha"]
    ) * hectares
    # 10-year NPV at 5% discount
    ecosystem_npv = ecosystem_annual * ((1 - (1 / 1.05**10)) / 0.05)

    timber_value = ECOSYSTEM_VALUES["timber_usd_per_ha"] * hectares
    total_loss   = carbon_value + ecosystem_npv + timber_value

    # Human-readable equivalents
    # 1 mature tree ≈ 21.7 kg CO₂/year → loss of stored carbon
    trees_eq = int(total_co2 * 1000 / 21.7)
    # Average car emits ~4.6 tonnes CO₂/year
    cars_eq  = int(total_co2 / 4.6)
    # Average US home emits ~7.5 tonnes CO₂/year
    homes_eq = int(total_co2 / 7.5)

    return CarbonImpact(
        hectares=round(hectares, 2),
        biome=biome,
        co2_per_ha=co2_per_ha,
        total_co2_tonnes=round(total_co2, 1),
        carbon_market=carbon_market,
        carbon_price_usd=carbon_price,
        carbon_value_usd=round(carbon_value, 0),
        ecosystem_value_usd=round(ecosystem_npv + timber_value, 0),
        total_economic_loss_usd=round(total_loss, 0),
        trees_equivalent=trees_eq,
        cars_equivalent=cars_eq,
        homes_equivalent=homes_eq,
    )


def format_currency(value: float) -> str:
    """Format large numbers as $1.2M, $340K, etc."""
    if value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.1f}K"
    return f"${value:.0f}"


def get_biome_names() -> list:
    return list(BIOME_COEFFICIENTS.keys())


def get_market_names() -> list:
    return list(CARBON_MARKETS.keys())
