import pandas as pd
import numpy as np
import os
from typing import Tuple

class ChurnFeaturePipeline:
    """
    Handles the ETL process: loads raw data, generates features,
    and prepares datasets for both Analytics (EDA/BI) and Machine Learning.
    """

    def __init__(self, input_path: str, output_dir: str):
        self.input_path = input_path
        self.output_dir = output_dir
        self.df = None

    def load_data(self) -> None:
        """Loads data from the raw CSV file."""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found at {self.input_path}")
        
        self.df = pd.read_csv(self.input_path)
        print(f"   [INFO] Data loaded. Shape: {self.df.shape}")

    def _generate_tenure_cohorts(self, months: int) -> str:
        """Helper function to categorize tenure for visualization."""
        if months < 12:
            return "0-1 Year"
        elif months < 24:
            return "1-2 Years"
        elif months < 48:
            return "2-4 Years"
        else:
            return "4+ Years"

    def engineering_step(self) -> None:
        """
        Creates new business-logic features.
        This enables richer storytelling (Yan Holtz style) later.
        """
        if self.df is None:
            raise ValueError("Dataframe is empty. Run load_data() first.")

        # 1. Tenure Cohorts (Critical for Churn Curves)
        self.df["tenure_group"] = self.df["tenure_months"].apply(self._generate_tenure_cohorts)

        # 2. Service Density (How many distinct services does the client have?)
        services = ["phone_service", "multiple_lines", "internet_service", "online_security", 
                   "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"]
        
        # We check which columns actually exist in our simulation to be safe
        existing_services = [col for col in services if col in self.df.columns]
        
        # Simple logic: if the value is not 'No' and not 'No internet service', it counts as a service
        self.df["total_services"] = self.df[existing_services].apply(
            lambda row: sum([1 for x in row if x not in ["No", "No internet service", "No phone service"]]), axis=1
        )

        # 3. Automatic Payment Flag (Correlates with retention)
        self.df["is_auto_payment"] = self.df["payment_method"].apply(
            lambda x: 1 if "automatic" in x else 0
        )

        print("   [INFO] Feature Engineering completed.")

    def export_analytical_dataset(self) -> None:
        """Saves the human-readable dataset for EDA and Power BI."""
        output_path = os.path.join(self.output_dir, "churn_data_analytical.csv")
        self.df.to_csv(output_path, index=False)
        print(f"   [SAVE] Analytical dataset saved to {output_path}")

    def prepare_and_export_ml_dataset(self) -> None:
        """
        Encodes categorical variables and saves the numerical dataset for ML.
        Uses One-Hot Encoding (OHE).
        """
        df_ml = self.df.copy()

        # Drop columns not useful for prediction
        cols_to_drop = ["customer_id", "tenure_group"] 
        df_ml = df_ml.drop(columns=[c for c in cols_to_drop if c in df_ml.columns])

        # Binary encoding for Target
        df_ml["churn"] = df_ml["churn"].apply(lambda x: 1 if x == "Yes" else 0)

        # One-Hot Encoding for remaining categorical variables
        df_ml = pd.get_dummies(df_ml, drop_first=True)

        output_path = os.path.join(self.output_dir, "churn_data_model_input.csv")
        df_ml.to_csv(output_path, index=False)
        print(f"   [SAVE] ML Input dataset saved to {output_path}")

    def run(self):
        """Orchestrates the pipeline."""
        print("--- Starting Feature Engineering Pipeline ---")
        self.load_data()
        self.engineering_step()
        self.export_analytical_dataset()
        self.prepare_and_export_ml_dataset()
        print("--- Pipeline Finished Successfully ---")

if __name__ == "__main__":
    # Define paths relative to the project root
    RAW_DATA = os.path.join("data", "raw", "telco_customer_churn_simulated.csv")
    PROCESSED_DIR = os.path.join("data", "processed")

    pipeline = ChurnFeaturePipeline(input_path=RAW_DATA, output_dir=PROCESSED_DIR)
    pipeline.run()
