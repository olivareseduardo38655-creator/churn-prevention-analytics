import pandas as pd
import numpy as np
from faker import Faker
import random
import os

# Configuración inicial (Semilla para reproducibilidad)
SEED = 42
NUM_CUSTOMERS = 5000  # Simulación de un mes de operación
Faker.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

fake = Faker()

def generate_customer_profiles(n: int) -> pd.DataFrame:
    """Genera datos demográficos y de servicios básicos."""
    data = []
    for _ in range(n):
        profile = {
            "customer_id": fake.unique.uuid4(),
            "gender": random.choice(["Male", "Female"]),
            "senior_citizen": random.choice([0, 1]),
            "partner": random.choice(["Yes", "No"]),
            "dependents": random.choice(["Yes", "No"]),
            "tenure_months": np.random.randint(1, 73),
            "phone_service": "Yes",
            "multiple_lines": random.choice(["No phone service", "No", "Yes"]),
            "internet_service": random.choice(["DSL", "Fiber optic", "No"]),
            "contract": random.choice(["Month-to-month", "One year", "Two year"]),
            "paperless_billing": random.choice(["Yes", "No"]),
            "payment_method": random.choice([
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ]),
            "monthly_charges": round(np.random.uniform(18.25, 118.75), 2)
        }
        data.append(profile)
    return pd.DataFrame(data)

def calculate_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula cargos totales con un poco de 'ruido' realista."""
    noise = np.random.normal(0, 5, size=len(df))
    df["total_charges"] = (df["tenure_months"] * df["monthly_charges"]) + noise
    # Evitar valores negativos y redondear
    df["total_charges"] = df["total_charges"].apply(lambda x: max(0, round(x, 2)))
    return df

def inject_churn_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inyecta lógica CAUSAL. 
    Esto es crucial: forzamos correlaciones para que el modelo tenga qué descubrir.
    """
    # Probabilidad base
    df["churn_probability"] = 0.20

    # Reglas de Aumento de Riesgo (Dolores de negocio)
    df.loc[df["contract"] == "Month-to-month", "churn_probability"] += 0.40
    df.loc[df["internet_service"] == "Fiber optic", "churn_probability"] += 0.15
    df.loc[df["payment_method"] == "Electronic check", "churn_probability"] += 0.10
    
    # Cliente nuevo pagando mucho = Fuga casi segura
    df.loc[(df["tenure_months"] < 12) & (df["monthly_charges"] > 70), "churn_probability"] += 0.25

    # Reglas de Fidelización (Reductores de riesgo)
    df.loc[df["contract"] == "Two year", "churn_probability"] -= 0.50
    df.loc[df["contract"] == "One year", "churn_probability"] -= 0.30
    df.loc[df["dependents"] == "Yes", "churn_probability"] -= 0.10
    df.loc[df["tenure_months"] > 60, "churn_probability"] -= 0.20

    # Limitar probabilidades entre 0 y 1
    df["churn_probability"] = df["churn_probability"].clip(0, 1)

    # Asignar etiqueta final basada en la probabilidad (Simulación de Monte Carlo simple)
    df["churn"] = df["churn_probability"].apply(lambda x: "Yes" if np.random.rand() < x else "No")
    
    # Eliminamos la probabilidad para que el modelo no haga "trampa"
    return df.drop(columns=["churn_probability"])

def main():
    print("--- 🏗️  Iniciando generación de datos sintéticos ---")
    
    # 1. Generar perfiles
    df_customers = generate_customer_profiles(NUM_CUSTOMERS)
    print(f"✅ {NUM_CUSTOMERS} perfiles generados.")
    
    # 2. Cálculos financieros
    df_financials = calculate_total_charges(df_customers)
    
    # 3. Inyectar patrones de fuga
    df_final = inject_churn_logic(df_financials)
    print("✅ Patrones de Churn inyectados (Causalidad establecida).")
    
    # 4. Guardar
    output_path = os.path.join("data", "raw", "telco_customer_churn_simulated.csv")
    df_final.to_csv(output_path, index=False)
    print(f"💾 Dataset guardado exitosamente en: {output_path}")

if __name__ == "__main__":
    main()
