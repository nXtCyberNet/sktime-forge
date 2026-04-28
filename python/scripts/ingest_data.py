import os
import argparse
import pandas as pd
import mlflow
from sktime.datasets import load_airline

def main():
    parser = argparse.ArgumentParser(description="Source data pipeline. Pulls datasets and logs to MLflow")
    parser.add_argument("--dataset", type=str, default="airline", help="Dataset name to ingest (e.g. airline)")
    args = parser.parse_args()

    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # 1. Fetch from source (sktime built-in)
    if args.dataset == "airline":
        print(f"[*] Fetching '{args.dataset}' dataset from sktime built-ins...")
        y = load_airline()
        df = pd.DataFrame({"y": y})
    else:
        print(f"[-] Dataset '{args.dataset}' not supported in this script yet.")
        return

    # 2. Extract metadata
    summary = {
        "n_observations": len(df),
        "source": "sktime.datasets",
        "dataset_name": args.dataset
    }
    
    # 3. Log to MLflow
    os.makedirs("data_cache", exist_ok=True)
    csv_path = f"data_cache/{args.dataset}.csv"
    df.to_csv(csv_path, index=False)
    
    experiment_name = "Data_Ingestion_Pipeline"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"Ingest_{args.dataset}"):
        print(f"[*] Logging data table to MLflow experiment '{experiment_name}'...")
        
        # Log summary parameters
        mlflow.log_params(summary)
        
        # Log the raw CSV file to the MLflow artifact store (which points to Minio behind the scenes)
        mlflow.log_artifact(csv_path, artifact_path="datasets")
        print(f"[*] Successfully saved artifact '{csv_path}' to MLflow.")
        
        try:
            # Modern MLflow dataset tracking (MLflow 2.4.0+)
            dataset = mlflow.data.from_pandas(df, source=f"sktime.datasets.{args.dataset}", name=args.dataset)
            mlflow.log_input(dataset, context="source_pipeline")
            print(f"[*] Registered MLflow Dataset Input for '{args.dataset}'")
        except Exception as e:
            print(f"[-] Could not register native MLflow Dataset Input (needs MLflow 2.4.0+): {e}")
            
    print(f"[*] Source pipeline completed for '{args.dataset}'!")

if __name__ == "__main__":
    main()