import os
import time
from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv
import logging
import mlflow.store.db.utils

def drop_all_tables(engine):
    inspector = inspect(engine)
    with engine.connect() as conn:
        conn.execute(text("SET session_replication_role = 'replica'"))
        
        tables = [
            "trace_request_metadata",
            "trace_tags",
            "trace_details",
            "trace_info",
            "input_tags",
            "inputs",
            "dataset_inputs",
            "dataset_tags",
            "datasets",
            "model_version_tags",
            "registered_model_tags",
            "model_versions",
            "registered_models",
            "model_aliases",
            "latest_metrics",
            "metrics",
            "params",
            "tags",
            "runs",
            "experiment_tags",
            "experiments",
            "alembic_version"
        ]
        
        for table in tables:
            try:
                conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                logging.info(f"Dropped table {table}")
            except Exception as e:
                logging.warning(f"Failed to drop table {table}: {e}")
        
        conn.execute(text("SET session_replication_role = 'origin'"))
        conn.commit()

def init_mlflow_db():
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
        
        drop_all_tables(engine)
        
        mlflow.store.db.utils._initialize_tables(engine)
        
        current_time = int(time.time() * 1000)  # Convert to milliseconds
        
        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO experiments (
                        experiment_id, 
                        name, 
                        artifact_location, 
                        lifecycle_stage,
                        creation_time,
                        last_update_time
                    ) VALUES (:id, :name, :location, :stage, :creation, :update)
                    """
                ),
                {
                    "id": 0,
                    "name": "Default",
                    "location": "mlruns/0",
                    "stage": "active",
                    "creation": current_time,
                    "update": current_time
                }
            )
            conn.commit()
        
        logging.info("MLflow database initialized successfully")
        
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    init_mlflow_db()