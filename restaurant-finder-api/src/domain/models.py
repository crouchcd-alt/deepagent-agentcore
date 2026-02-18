"""
Structured output models for the restaurant finder service.

These Pydantic models define the schema for structured responses from agents,
ensuring consistent and typed data flows through the system.
"""

from enum import Enum
from pydantic import BaseModel, Field


class PriceRange(str, Enum):
    """Price range categories for restaurants."""
    BUDGET = "$"
    MODERATE = "$$"
    UPSCALE = "$$$"
    FINE_DINING = "$$$$"


# Mapping from price string notation to PriceRange enum
PRICE_RANGE_MAP = {
    "$": PriceRange.BUDGET,
    "$$": PriceRange.MODERATE,
    "$$$": PriceRange.UPSCALE,
    "$$$$": PriceRange.FINE_DINING,
}


class Restaurant(BaseModel):
    """Structured representation of a restaurant."""

    name: str = Field(
        description="The name of the restaurant"
    )
    cuisine_type: str = Field(
        description="The type of cuisine (e.g., Italian, Japanese, Indian)"
    )
    rating: float = Field(
        ge=0.0,
        le=5.0,
        description="Rating out of 5 stars"
    )
    review_count: int | None = Field(
        default=None,
        description="Number of reviews"
    )
    price_range: PriceRange = Field(
        description="Price range category"
    )
    address: str | None = Field(
        default=None,
        description="Street address of the restaurant"
    )
    city: str | None = Field(
        default=None,
        description="City where the restaurant is located"
    )
    phone: str | None = Field(
        default=None,
        description="Contact phone number"
    )
    website: str | None = Field(
        default=None,
        description="Restaurant website URL"
    )
    features: list[str] = Field(
        default_factory=list,
        description="List of features (e.g., outdoor seating, parking, vegetarian options)"
    )
    dietary_options: list[str] = Field(
        default_factory=list,
        description="Dietary accommodations (e.g., vegetarian, vegan, gluten-free, halal)"
    )
    operating_hours: str | None = Field(
        default=None,
        description="Operating hours or schedule"
    )
    distance: str | None = Field(
        default=None,
        description="Distance from search location"
    )
    reservation_available: bool | None = Field(
        default=None,
        description="Whether reservations are available"
    )


class RestaurantSearchResult(BaseModel):
    """Structured output for restaurant search results."""

    query: str = Field(
        description="The original search query"
    )
    total_results: int = Field(
        description="Total number of restaurants found"
    )
    restaurants: list[Restaurant] = Field(
        description="List of restaurants matching the search criteria"
    )
    search_location: str | None = Field(
        default=None,
        description="The location used for the search"
    )
    search_filters: dict[str, str] = Field(
        default_factory=dict,
        description="Filters applied to the search (cuisine, price, etc.)"
    )
    data_source: str = Field(
        default="searchapi",
        description="Source of the data (e.g., 'searchapi', 'browser')"
    )
    notes: str | None = Field(
        default=None,
        description="Additional notes or caveats about the search results"
    )


