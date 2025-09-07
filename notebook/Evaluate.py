import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Load dataset
data = pd.read_csv(r"c:\diabetes_analysis_project\datasets\diabetes_data_ml.csv")   # adjust path/filename
print(f'model features length {len(data.columns)}')
# 2. Split features/target
X = data.drop("readmitted", axis=1)   # replace "readmitted" with your actual target column
y = data["readmitted"]

# 3. Train/test split (same random_state used during training)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Load trained model
model = joblib.load(r"c:\diabetes_analysis_project\notebook\diabetes_readmission.pkl")   # adjust path
# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")
