from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "india_housing_prices.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MLFLOW_EXPERIMENT = "RealEstateInvestment"
SEED = 42
DEFAULT_GROWTH_RATE = 0.05
FUTURE_YEARS = 5

