from __future__ import annotations

from app.data.local_loader import build_data_loader
from app.mcp.client import MCPClient


def test_airline_seasonality_detection_is_strong_monthly() -> None:
    """Airline is a canonical monthly seasonal series with period ~12."""
    client = MCPClient(data_loader=build_data_loader(None))

    result = client.detect_seasonality("airline", freq="M")

    assert result["period"] == 12
    assert result["seasonality_class"] == "strong"
    assert result["strength"] >= 0.6
    assert result["confidence"] in {"medium", "high"}
