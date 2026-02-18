"""
Main entry point for the Restaurant Finder Agent.

This module serves as the container entry point when running with
opentelemetry-instrument for automatic tracing and observability.

Usage:
    opentelemetry-instrument python -m src.main
"""

from src.infrastructure.api import app

if __name__ == "__main__":
    app.run()
