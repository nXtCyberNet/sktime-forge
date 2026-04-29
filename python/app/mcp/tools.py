"""
OpenAI-compatible tool schemas for MCPClient methods.
Pass these to the LLM so it can call MCP tools during reasoning.
"""

MCP_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "profile_dataset",
            "description": "Full diagnostic: stationarity, seasonality, structural break, complexity budget in one call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "freq": {"type": "string", "description": "Optional pandas freq string e.g. 'M', 'D'"}
                },
                "required": ["dataset_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_structural_break",
            "description": "CUSUM-based structural break detection. Use when LLM needs to verify changepoints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"}
                },
                "required": ["dataset_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_seasonality",
            "description": "Autocorrelation-based seasonality detection. Returns period, strength, confidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "freq": {"type": "string"}
                },
                "required": ["dataset_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_dataset_history",
            "description": "Production memory — past model failures, drift events, what was tried before.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"}
                },
                "required": ["dataset_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_training_cost",
            "description": "Estimate fit time and cost for a model class before committing to training.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "model_class": {"type": "string"},
                    "seasonality_period": {"type": "integer", "default": 1}
                },
                "required": ["dataset_id", "model_class"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_stationarity_test",
            "description": "ADF + KPSS stationarity tests. Returns conclusion, p-values, recommendation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"}
                },
                "required": ["dataset_id"]
            }
        }
    },
]

# Router — maps tool name → MCPClient method
def dispatch_tool(mcp_client, tool_name: str, args: dict):
    method = getattr(mcp_client, tool_name, None)
    if method is None:
        raise ValueError(f"Unknown MCP tool: {tool_name}")
    return method(**args)