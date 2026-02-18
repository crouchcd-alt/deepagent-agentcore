# Wrapper to expose the app at the root level for agentcore CLI
from src.infrastructure.api import app

__all__ = ["app"]

if __name__ == "__main__":
    app.run()
