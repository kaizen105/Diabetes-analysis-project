import unittest
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class TestLogRegPipeline(unittest.TestCase):
    def test_logreg_pipeline(self):
        # Load the pipeline
        model = joblib.load(r"c:\diabetes_analysis_project\notebook\diabetes_readmission.pkl")

        # Check the outer object is a Pipeline
        self.assertIsInstance(model, Pipeline)

        # Extract the Logistic Regression step
        lr = model.named_steps.get("lr", None)
        self.assertIsNotNone(lr, "Pipeline does not contain a step named 'lr'")
        self.assertIsInstance(lr, LogisticRegression)

        # Check feature count (71 features)
        self.assertEqual(lr.coef_.shape[1], 70)

        # Check binary classification
        self.assertEqual(len(lr.classes_), 2)

        # Check intercept
        self.assertEqual(len(lr.intercept_), 1)

if __name__ == "__main__":
    unittest.main()
