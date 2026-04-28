import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    mlflow_db = root / "mlflow.db"
    artifact_root = root / "mlflow-artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    backend_store_uri = f"sqlite:///{mlflow_db.resolve().as_posix()}"
    artifact_root_uri = artifact_root.resolve().as_uri()

    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        "5000",
        "--backend-store-uri",
        backend_store_uri,
        "--default-artifact-root",
        artifact_root_uri,
    ]

    print("Starting local MLflow server with:")
    print(" ".join(cmd))
    print("Tracking URI: http://127.0.0.1:5000")
    print("Artifact root:", artifact_root.resolve())
    print("Press Ctrl+C to stop.")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("Local MLflow server stopped.")


if __name__ == "__main__":
    main()
