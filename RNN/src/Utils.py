from pathlib import Path
import HyperParameters
import Utils as U

#Dataset location information 
PROJECT_DIR = Path(__file__).parent.parent
CLEAN_DATA_FOLDER = (PROJECT_DIR / 'data_clean').resolve()
RAW_DATA_FOLDER = (PROJECT_DIR / 'data_raw').resolve()
TEST_DATA_FOLDER = (PROJECT_DIR / 'data_test').resolve()
MODEL_FOLDER = (PROJECT_DIR / 'Model').resolve()

data_raw = (U.RAW_DATA_FOLDER / 'finance-charts-apple.csv').resolve()

X1 = (U.CLEAN_DATA_FOLDER / 'X1.pt').resolve()
X2 = (U.CLEAN_DATA_FOLDER / 'X2.pt').resolve()
y = (U.CLEAN_DATA_FOLDER / 'y.pt').resolve()
model_predict = (U.CLEAN_DATA_FOLDER / 'model_predict.pt').resolve()
