import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os
import joblib

# Configuración
SEED = 42
pd.options.mode.chained_assignment = None  # Silenciar advertencias menores

class ChurnPredictor:
    """
    Train a Random Forest model and generate Explainable AI (XAI) outputs
    specifically formatted for Power BI consumption.
    """
    
    def __init__(self, input_ml_path: str, input_analytical_path: str, output_dir: str):
        self.input_ml_path = input_ml_path
        self.input_analytical_path = input_analytical_path
        self.output_dir = output_dir
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Loads both the numerical (ML) and human-readable (Analytical) datasets."""
        print("   [LOAD] Cargando datasets...")
        self.df_ml = pd.read_csv(self.input_ml_path)
        self.df_analytical = pd.read_csv(self.input_analytical_path)

    def train_model(self):
        """Trains the Random Forest model."""
        print("   [TRAIN] Entrenando modelo Random Forest...")
        
        X = self.df_ml.drop(columns=["churn"])
        y = self.df_ml["churn"]
        
        # Split 80/20
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
        
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED)
        self.model.fit(self.X_train, self.y_train)
        
        # Validación rápida
        preds = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        print(f"   [METRICS] Accuracy del modelo: {acc:.2%}")

    def generate_predictions_and_explanations(self):
        """
        1. Predicts Churn Probability for ALL customers.
        2. Uses SHAP values to explain WHY a specific customer is at risk.
        3. Merges everything into the 'Analytical' dataset for Power BI.
        """
        print("   [XAI] Generando explicaciones con SHAP (esto puede tardar unos segundos)...")
        
        # Predecir probabilidades en todo el dataset
        X_all = self.df_ml.drop(columns=["churn"])
        probs = self.model.predict_proba(X_all)[:, 1] # Probabilidad de clase 1 (Yes)
        
        # Generar valores SHAP (Explicabilidad)
        explainer = shap.TreeExplainer(self.model)
        
        # Corrección de Compatibilidad SHAP (check_additivity=False para evitar errores de redondeo)
        shap_values = explainer.shap_values(X_all, check_additivity=False)
        
        # --- FIX PARA VERSIONES NUEVAS DE SHAP ---
        # Detectamos si devuelve una lista (versión vieja) o un array 3D (versión nueva)
        if isinstance(shap_values, list):
            # Formato lista: [Matriz_Clase0, Matriz_Clase1]
            shap_values_churn = shap_values[1]
        else:
            # Formato Array: (Muestras, Features, Clases)
            # Queremos todas las muestras, todas las features, clase 1 (índice 1)
            shap_values_churn = shap_values[:, :, 1]
        
        # --- Lógica de Negocio para Power BI ---
        feature_names = X_all.columns
        top_reasons = []
        
        # Iteramos con cuidado
        print("   [XAI] Procesando razones principales de fuga...")
        for i in range(len(X_all)):
            # Obtener los valores SHAP de este cliente
            client_shap = shap_values_churn[i]
            
            # Encontrar el índice de la característica con mayor impacto positivo (que empuja al churn)
            max_impact_idx = np.argmax(client_shap)
            
            # Si el impacto es positivo (aumenta riesgo), guardamos la razón
            if client_shap[max_impact_idx] > 0:
                top_reasons.append(feature_names[max_impact_idx])
            else:
                top_reasons.append("Bajo Riesgo / N/A")

        # --- Integración Final ---
        df_final = self.df_analytical.copy()
        df_final["Probability_Churn"] = np.round(probs, 4)
        
        # Segmentación de riesgo
        df_final["Risk_Segment"] = pd.cut(df_final["Probability_Churn"], 
                                          bins=[-1, 0.3, 0.7, 1.0], 
                                          labels=["Low Risk", "Medium Risk", "High Risk"])
        
        df_final["Main_Churn_Reason"] = top_reasons

        # Limpieza de nombres para Power BI
        df_final["Main_Churn_Reason"] = df_final["Main_Churn_Reason"].str.replace("_", " ").str.title()
        
        self.df_final = df_final

    def save_for_powerbi(self):
        """Saves the Gold dataset."""
        output_path = os.path.join(self.output_dir, "PROJECT_GOLD_DATASET_POWERBI.csv")
        self.df_final.to_csv(output_path, index=False)
        print(f"   [SUCCESS] Archivo ORO generado para Power BI: {output_path}")

    def run(self):
        self.load_data()
        self.train_model()
        self.generate_predictions_and_explanations()
        self.save_for_powerbi()

if __name__ == "__main__":
    ML_DATA = os.path.join("data", "processed", "churn_data_model_input.csv")
    ANALYTICAL_DATA = os.path.join("data", "processed", "churn_data_analytical.csv")
    OUTPUT_DIR = os.path.join("data", "processed")

    predictor = ChurnPredictor(ML_DATA, ANALYTICAL_DATA, OUTPUT_DIR)
    predictor.run()
