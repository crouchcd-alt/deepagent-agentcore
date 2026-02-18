import json
import os
import urllib.request
import urllib.parse
import urllib.error
from typing import Any, Dict, List

import boto3
from botocore.exceptions import ClientError

SEARCH_BASE_URL = "https://www.searchapi.io/api/v1/search"
_cached_api_key: str | None = None


def _get_search_api_key() -> str:
    """
    Retrieve the search API key from AWS Secrets Manager.

    Cached after first retrieval to avoid repeated calls during warm starts.
    """
    global _cached_api_key

    if _cached_api_key is not None:
        return _cached_api_key

    secret_name = os.environ.get("SEARCH_SECRET_NAME")

    if not secret_name:
        raise RuntimeError(
            "SEARCH_SECRET_NAME environment variable not set. "
            "Configure the secret in AWS Secrets Manager."
        )

    try:
        client = boto3.client("secretsmanager")
        response = client.get_secret_value(SecretId=secret_name)

        secret_string = response["SecretString"]

        try:
            secret_data = json.loads(secret_string)
            if isinstance(secret_data, dict):
                _cached_api_key = (
                    secret_data.get("api_key")
                    or secret_data.get("key")
                    or secret_string
                )
            else:
                _cached_api_key = secret_string
        except json.JSONDecodeError:
            # Plain text secret
            _cached_api_key = secret_string

        return _cached_api_key

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        raise RuntimeError(
            f"Failed to retrieve secret from Secrets Manager: {error_code}. "
            f"Ensure '{secret_name}' exists and Lambda has read access."
        )


def lambda_handler(event, context):
    """Route tool invocations from the AgentCore Gateway."""
    try:
        extended_name = context.client_context.custom.get("bedrockAgentCoreToolName")
        tool_name = None

        if extended_name and "___" in extended_name:
            tool_name = extended_name.split("___", 1)[1]

        if not tool_name:
            return _response(400, {"error": "Missing tool name"})

        if tool_name == "search_restaurants":
            result = search_restaurants(event)
            return _response(200, {"result": result})
        else:
            return _response(400, {"error": f"Unknown tool '{tool_name}'"})

    except Exception as e:
        return _response(500, {"system_error": str(e)})


def _response(status_code: int, body: Dict[str, Any]):
    return {"statusCode": status_code, "body": json.dumps(body)}


def _search_local(query: str, location: str = "", num_results: int = 10) -> Dict[str, Any]:
    """Search using google_local engine for structured local business data."""
    api_key = _get_search_api_key()

    params = {
        "api_key": api_key,
        "engine": "google_local",
        "q": query,
        "num": str(num_results),
    }

    if location and location.strip():
        params["location"] = location.strip()

    url = f"{SEARCH_BASE_URL}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"Search HTTP error {e.code}: {error_body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Search connection error: {e.reason}")


def _search_web(query: str, num_results: int = 10) -> Dict[str, Any]:
    """Fallback web search using google engine."""
    api_key = _get_search_api_key()

    params = {
        "api_key": api_key,
        "engine": "google",
        "q": query,
        "num": str(num_results),
    }

    url = f"{SEARCH_BASE_URL}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"Search HTTP error {e.code}: {error_body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Search connection error: {e.reason}")


def _build_search_query(
    query: str = "",
    cuisine: str = "",
    location: str = "",
    price_range: str = "",
    dietary_restrictions: List[str] = None,
) -> str:
    """Build a composite search query from structured parameters."""
    parts = []

    if query and query.strip():
        parts.append(query.strip())

    if cuisine and cuisine.strip():
        parts.append(f"{cuisine.strip()} restaurants")
    elif not query:
        parts.append("restaurants")

    if location and location.strip():
        parts.append(f"in {location.strip()}")

    price_descriptions = {
        "$": "budget-friendly cheap",
        "$$": "moderate mid-range",
        "$$$": "upscale high-end",
        "$$$$": "fine dining luxury",
    }
    if price_range and price_range in price_descriptions:
        parts.append(price_descriptions[price_range])

    if dietary_restrictions:
        if isinstance(dietary_restrictions, str):
            dietary_restrictions = [d.strip() for d in dietary_restrictions.split(",") if d.strip()]
        if dietary_restrictions:
            parts.append(" ".join(dietary_restrictions))

    return " ".join(parts)


def _parse_local_results(
    api_response: Dict[str, Any],
    location: str,
    cuisine: str,
    price_range: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """Parse google_local response into structured restaurant objects."""
    restaurants = []
    local_results = api_response.get("local_results", [])

    for idx, result in enumerate(local_results[:limit]):
        name = result.get("title") or result.get("name", f"Restaurant {idx + 1}")

        rating = result.get("rating", 0.0)
        if isinstance(rating, str):
            try:
                rating = float(rating)
            except ValueError:
                rating = 0.0

        reviews = result.get("reviews", 0)
        if isinstance(reviews, str):
            reviews = int("".join(filter(str.isdigit, reviews)) or "0")

        type_info = result.get("type", result.get("types", ""))
        if isinstance(type_info, list):
            cuisine_type = ", ".join(type_info[:3]) if type_info else cuisine or "Restaurant"
        else:
            cuisine_type = type_info if type_info else cuisine or "Restaurant"

        address = result.get("address", "")
        service_options = result.get("service_options", {})
        features = []
        if isinstance(service_options, dict):
            if service_options.get("dine_in"):
                features.append("Dine-in")
            if service_options.get("takeout"):
                features.append("Takeout")
            if service_options.get("delivery"):
                features.append("Delivery")
        elif isinstance(service_options, list):
            features = service_options

        hours = result.get("hours", result.get("operating_hours", ""))
        if isinstance(hours, dict):
            hours = hours.get("today", "")

        restaurant = {
            "name": name,
            "cuisine_type": cuisine_type,
            "rating": round(float(rating), 1) if rating else 0.0,
            "review_count": int(reviews) if reviews else 0,
            "price_range": result.get("price", price_range or "$$"),
            "address": address[:200] if address else "",
            "city": location.title() if location else "",
            "neighborhood": result.get("neighborhood", ""),
            "features": features,
            "dietary_options": [],
            "operating_hours": hours if isinstance(hours, str) else "",
            "reservation_available": "reservations" in str(service_options).lower(),
            "phone": result.get("phone", ""),
            "website": result.get("website", result.get("link", "")),
            "thumbnail": result.get("thumbnail", ""),
            "gps_coordinates": result.get("gps_coordinates", {}),
            "place_id": result.get("place_id", ""),
            "source": "google_local",
        }
        restaurants.append(restaurant)

    return restaurants


def _parse_web_results(
    api_response: Dict[str, Any],
    location: str,
    cuisine: str,
    price_range: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """Parse google web search response into restaurant objects (fallback)."""
    restaurants = []
    organic_results = api_response.get("organic_results", [])

    for idx, result in enumerate(organic_results[:limit]):
        title = result.get("title", f"Restaurant {idx + 1}")
        snippet = result.get("snippet", "")
        rating = 0.0
        reviews = 0

        restaurant = {
            "name": title,
            "cuisine_type": cuisine or "Restaurant",
            "rating": rating,
            "review_count": reviews,
            "price_range": price_range or "$$",
            "address": snippet[:200] if snippet else "",
            "city": location.title() if location else "",
            "neighborhood": "",
            "features": [],
            "dietary_options": [],
            "operating_hours": "",
            "reservation_available": False,
            "phone": "",
            "website": result.get("link", ""),
            "source": "web_search",
        }
        restaurants.append(restaurant)

    return restaurants


def search_restaurants(event: Dict[str, Any]) -> Dict[str, Any]:
    """Search for restaurants. Tries local search first, falls back to web."""
    query = event.get("query", "").strip()
    cuisine = event.get("cuisine", "").strip()
    location = event.get("location", "").strip()
    price_range = event.get("price_range", "$$")
    dietary_restrictions = event.get("dietary_restrictions", [])
    limit = min(int(event.get("limit", 5)), 10)

    if isinstance(dietary_restrictions, str):
        dietary_restrictions = [d.strip() for d in dietary_restrictions.split(",") if d.strip()]

    search_query = _build_search_query(
        query=query,
        cuisine=cuisine,
        location=location,
        price_range=price_range,
        dietary_restrictions=dietary_restrictions,
    )

    restaurants = []
    data_source = "google_local"
    error_message = None

    try:
        api_response = _search_local(
            query=search_query,
            location=location,
            num_results=limit * 2,
        )

        restaurants = _parse_local_results(
            api_response=api_response,
            location=location,
            cuisine=cuisine,
            price_range=price_range,
            limit=limit,
        )

        if not restaurants:
            data_source = "web_search"
            api_response = _search_web(search_query, num_results=limit * 2)

            restaurants = _parse_web_results(
                api_response=api_response,
                location=location,
                cuisine=cuisine,
                price_range=price_range,
                limit=limit,
            )

    except Exception as e:
        error_message = str(e)
        try:
            data_source = "web_search"
            api_response = _search_web(search_query, num_results=limit * 2)

            restaurants = _parse_web_results(
                api_response=api_response,
                location=location,
                cuisine=cuisine,
                price_range=price_range,
                limit=limit,
            )
        except Exception as fallback_error:
            error_message = f"Local: {error_message}, Web: {str(fallback_error)}"

    restaurants.sort(key=lambda x: x.get("rating", 0), reverse=True)

    result = {
        "restaurants": restaurants,
        "total_found": len(restaurants),
        "search_params": {
            "query": query,
            "cuisine": cuisine or "any",
            "location": location or "any",
            "price_range": price_range,
            "dietary_restrictions": dietary_restrictions,
            "limit": limit,
        },
        "search_query_used": search_query,
        "data_source": data_source,
        "message": f"Found {len(restaurants)} restaurants via {data_source}.",
    }

    if error_message and not restaurants:
        result["error"] = error_message
        result["message"] = f"Search failed: {error_message}"

    return result

