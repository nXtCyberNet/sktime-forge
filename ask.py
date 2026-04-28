import argparse
import sys
import httpx
import json

def main():
    parser = argparse.ArgumentParser(description="sktime-agentic NLP CLI tool (Agentic pipeline mode)")
    parser.add_argument("prompt", type=str, help="English prompt for forecasting")
    args = parser.parse_args()

    print(f"[*] Sending natural language to Python Orchestrator Agent Router...")
    
    try:
        # Hit the new /chat endpoint on the Python Worker (default port 8000)
        # Note: If Go gateway starts proxying /chat we can update to 8080 later
        resp = httpx.post(
            "http://localhost:8000/chat", 
            json={"query": args.prompt}, 
            timeout=120.0
        )
    except httpx.RequestError as e:
        print(f"Network error querying Agent: {e}")
        sys.exit(1)

    if resp.status_code == 200:
        print("\n=== Agent Pipeline Completed successfully! ===")
        print("\n=== Forecast Result ===")
        print(json.dumps(resp.json(), indent=2))
        print("=======================\n")
    else:
        print(f"Agent Pipeline Error ({resp.status_code}): {resp.text}")

if __name__ == "__main__":
    main()
