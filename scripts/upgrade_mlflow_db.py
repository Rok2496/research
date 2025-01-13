# scripts/upgrade_mlflow_db.py
import os
from dotenv import load_dotenv
import mlflow.store.db.utils
from sqlalchemy import create_engine

def upgrade_mlflow_db():
    load_dotenv()
    
    db_url = (
        f"postgresql://{os.getenv('POSTGRES_USER')}:"
        f"{os.getenv('POSTGRES_PASSWORD')}@"
        f"{os.getenv('POSTGRES_HOST')}:"
        f"{os.getenv('POSTGRES_PORT')}/"
        f"{os.getenv('POSTGRES_DB')}"
    )
    
    try:
        engine = create_engine(db_url)
        mlflow.store.db.utils._upgrade_db(engine)
    except Exception as e:
        raise RuntimeError(f"Failed to upgrade MLflow database: {e}")

if __name__ == "__main__":
    upgrade_mlflow_db()