# src/data_loader.py
import os
import shutil
from pathlib import Path
from src.utils import setup_logger

logger = setup_logger("DataLoader")

def check_data_exists(data_dir):
    """Check if data directory exists and contains files."""
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.info(f"Data directory {data_dir} does not exist.")
        return False
    
    # Check for train and test subdirectories with actual CSV files
    train_dir = data_path / "train"
    test_dir = data_path / "test"
    
    # Check if directories exist and contain CSV files
    train_exists = train_dir.exists() and len(list(train_dir.glob("*.csv"))) > 0
    test_exists = test_dir.exists() and len(list(test_dir.glob("*.csv"))) > 0
    
    # Check for submission file (could be named sample_submission.csv or submission.csv)
    submission_exists = (
        (data_path / "sample_submission.csv").exists() or 
        (data_path / "submission.csv").exists()
    )
    
    if train_exists and test_exists and submission_exists:
        train_count = len(list(train_dir.glob("*.csv")))
        test_count = len(list(test_dir.glob("*.csv")))
        logger.info(f"✓ Data found at {data_dir} (train: {train_count} wells, test: {test_count} wells)")
        return True
    
    logger.warning(f"Data incomplete at {data_dir}: train={train_exists} ({len(list(train_dir.glob('*.csv'))) if train_dir.exists() else 0} files), test={test_exists} ({len(list(test_dir.glob('*.csv'))) if test_dir.exists() else 0} files), submission={submission_exists}")
    return False

def download_competition_data(data_dir, competition_name='rogii-wellbore-geology-prediction'):
    """
    Download competition data using kagglehub.
    
    Args:
        data_dir: Target directory for the data
        competition_name: Kaggle competition name
    
    Returns:
        Path to the data directory
    
    Exits with error if:
        - KAGGLE_API_TOKEN is not set
        - Download fails
    """
    data_path = Path(data_dir)
    
    # Check if data already exists
    if check_data_exists(data_dir):
        logger.info(f"Data already exists at {data_dir}. Skipping download.")
        return str(data_path)
    
    # Check for API token
    token = os.environ.get('KAGGLE_API_TOKEN')
    if not token:
        logger.error("KAGGLE_API_TOKEN environment variable is not set.")
        logger.error("Please set it with: export KAGGLE_API_TOKEN='your_token_here'")
        logger.error("Or ensure kaggle_token.txt is loaded into the environment.")
        exit(1)
    
    logger.info(f"KAGGLE_API_TOKEN found. Downloading competition: {competition_name}")
    
    try:
        import kagglehub
        
        # Download the competition data
        logger.info("Starting download...")
        downloaded_path = kagglehub.competition_download(competition_name)
        logger.info(f"Downloaded to: {downloaded_path}")
        
        # kagglehub downloads to a cache directory, we need to copy to our data_dir
        downloaded_path = Path(downloaded_path)
        
        # Create target directory if it doesn't exist
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Copy files from downloaded location to our data_dir
        logger.info(f"Copying files to {data_dir}...")
        
        for item in downloaded_path.iterdir():
            dest = data_path / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        
        logger.info(f"Data successfully copied to {data_dir}")
        return str(data_path)
        
    except ImportError:
        logger.error("kagglehub is not installed. Please install it with: pip install kagglehub")
        exit(1)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.error("Make sure your KAGGLE_API_TOKEN is valid and you have accepted the competition rules.")
        exit(1)
