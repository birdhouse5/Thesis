# simple_mlflow_setup.py - Minimal MLflow setup for existing infrastructure

import os
import mlflow
import logging

logger = logging.getLogger(__name__)

def setup_mlflow():
    """
    Simple MLflow setup using existing environment variables.
    Assumes MLFLOW_TRACKING_URI, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, 
    and MLFLOW_S3_ENDPOINT_URL are already set.
    """
    
    # Check required environment variables
    required_vars = [
        'MLFLOW_TRACKING_URI',
        'AWS_ACCESS_KEY_ID', 
        'AWS_SECRET_ACCESS_KEY',
        'MLFLOW_S3_ENDPOINT_URL'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        return False
    
    # Set MLflow tracking URI
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set TLS ignore for MinIO (assuming it's HTTP not HTTPS)
    os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'
    
    logger.info(f"MLflow configured:")
    logger.info(f"  Tracking URI: {tracking_uri}")
    logger.info(f"  S3 Endpoint: {os.getenv('MLFLOW_S3_ENDPOINT_URL')}")
    
    # Test connection
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        logger.info(f"✅ MLflow connected. Found {len(experiments)} experiments.")
        return True
    except Exception as e:
        logger.error(f"❌ MLflow connection failed: {e}")
        return False

def log_essential_artifacts(model_dict, config_dict, experiment_name):
    """
    Log only the essential artifacts to MinIO.
    
    Args:
        model_dict: Dict with 'policy' and optionally 'encoder' PyTorch models
        config_dict: Experiment configuration dictionary
        experiment_name: Name of the experiment
    """
    
    # Log models (these go to MinIO automatically)
    if 'policy' in model_dict and model_dict['policy'] is not None:
        mlflow.pytorch.log_model(
            model_dict['policy'], 
            "policy_model",
            registered_model_name=f"{experiment_name}_policy"
        )
        logger.info("✅ Policy model logged to MinIO")
    
    if 'encoder' in model_dict and model_dict['encoder'] is not None:
        mlflow.pytorch.log_model(
            model_dict['encoder'],
            "encoder_model",
            registered_model_name=f"{experiment_name}_encoder" 
        )
        logger.info("✅ Encoder model logged to MinIO")
    
    # Log config as JSON artifact
    import json
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_dict, f, indent=2, default=str)
        config_file = f.name
    
    mlflow.log_artifact(config_file, "config")
    os.unlink(config_file)  # Clean up temp file
    logger.info("✅ Configuration logged to MinIO")

def log_training_checkpoint(trainer_state, episode):
    """
    Log training checkpoint to MinIO.
    
    Args:
        trainer_state: Dictionary with trainer state (from trainer.get_state())
        episode: Current episode number
    """
    
    import json
    import tempfile
    import torch
    
    # Create checkpoint data
    checkpoint_data = {
        'episode': episode,
        'trainer_state': trainer_state,
        'timestamp': str(mlflow.utils.time.get_current_time_millis())
    }
    
    # Save as temporary file and log as artifact
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(checkpoint_data, f, indent=2, default=str)
        checkpoint_file = f.name
    
    mlflow.log_artifact(checkpoint_file, f"checkpoints/episode_{episode}")
    os.unlink(checkpoint_file)
    
    logger.info(f"✅ Training checkpoint logged for episode {episode}")

# Test function
def test_minio_connection():
    """Test if we can actually write to MinIO"""
    
    if not setup_mlflow():
        return False
    
    try:
        # Start a test run
        with mlflow.start_run(run_name="minio_connection_test"):
            
            # Test logging a simple metric
            mlflow.log_metric("test_metric", 42.0)
            
            # Test logging a simple artifact
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("This is a test artifact for MinIO")
                test_file = f.name
            
            mlflow.log_artifact(test_file, "test_artifacts")
            os.unlink(test_file)
            
            logger.info("✅ MinIO artifact test successful!")
            return True
            
    except Exception as e:
        logger.error(f"❌ MinIO test failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing MLflow + MinIO connection...")
    success = test_minio_connection()
    
    if success:
        print("🎉 Everything is working! Your experiments will store artifacts in MinIO.")
    else:
        print("⚠️  Connection failed. Check your environment variables and MinIO setup.")