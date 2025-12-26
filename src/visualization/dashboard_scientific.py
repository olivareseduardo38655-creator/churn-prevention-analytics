import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Informe Científico de Retención", page_icon="📈")

# --- INYECCIÓN DE CSS (TU ESTILO ACADÉMICO) ---
st.markdown("""
<style>
    /* Tipografía Serif elegante para todo el reporte */
    body, .stMarkdown, p, h1, h2, h3, li {
        font-family: 'Georgia', 'Garamond', serif !important;
        color: #1a1a1a;
    }
    
    /* Títulos sobrios */
    h1 { font-size: 2.5em; font-weight: bold; border-bottom: 2px solid #333; padding-bottom: 10px; }
    h2 { font-size: 1.8em; color: #444; margin-top: 30px; }
    h3 { font-size: 1.4em; color: #666; font-style: italic; }

    /* Observation Box (Discusión de Resultados) */
    .observation-box {
        background-color: #ffffff;
        border-left: 4px solid #457b9d; /* Azul acero sobrio */
        padding: 15px;
        margin: 20px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        font-size: 16px;
        line-height: 1.6;
        text-align: justify;
    }

    /* Prescription Box (Acciones Estratégicas) */
    .prescription-box {
        background-color: #f4fcf4;
        border-left: 4px solid #2a9d8f; /* Verde sobrio */
        padding: 15px;
        margin: 20px 0;
        font-size: 16px;
        line-height: 1.6;
    }

    /* Estilo para la Matriz Lógica */
    .logic-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-family: 'Georgia', serif;
    }
    .logic-table th {
        background-color: #f0f0f0;
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
        font-weight: bold;
    }
    .logic-table td {
        border: 1px solid #ddd;
        padding: 12px;
        vertical-align: top;
    }
    
    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    # Ruta relativa ajustada a la estructura del proyecto
    path = os.path.join("data", "processed", "churn_data_analytical.csv")
    if not os.path.exists(path):
        st.error(f"No se encontró el archivo en: {path}. Ejecuta primero el pipeline de ETL.")
        return None
    return pd.read_csv(path)

df = load_data()

if df is not None:
    # --- I. PLANTEAMIENTO DEL PROBLEMA ---
    st.markdown("# I. Estudio Analítico de Fuga de Clientes (Churn)")
    st.markdown("""
    **Resumen Ejecutivo:** El presente informe analiza los determinantes microeconómicos que propician la cancelación de contratos 
    en la cartera de clientes. El objetivo es aislar variables causales y proponer una estrategia de retención basada en evidencia.
    """)

    # --- II. METODOLOGÍA ---
    st.markdown("## II. Metodología y Diseño Experimental")
    st.markdown(f"""
    Se analizó una muestra de **{len(df):,} clientes**, segmentados por antigüedad, tipo de contrato y servicios contratados. 
    Se aplicaron técnicas de estadística descriptiva para identificar correlaciones no lineales entre la facturación mensual y la probabilidad de deserción.
    """)

    # --- III. ANÁLISIS EMPÍRICO ---
    st.markdown("## III. Análisis Empírico (Diagnóstico)")
    
    tab1, tab2, tab3 = st.tabs(["Facturación vs Fuga", "Riesgo Contractual", "Densidad de Servicios"])

    with tab1:
        st.markdown("### Distribución de Cargos Mensuales")
        # Gráfico Violin (Plotly) - Estilo Yan Holtz
        fig_violin = go.Figure()
        
        # Separar datos
        churn_yes = df[df["churn"] == "Yes"]["monthly_charges"]
        churn_no = df[df["churn"] == "No"]["monthly_charges"]

        fig_violin.add_trace(go.Violin(x=df["churn"][df["churn"] == "Yes"], 
                                       y=churn_yes, 
                                       name='Fugado', 
                                       line_color='#e63946', 
                                       side='positive'))
        
        fig_violin.add_trace(go.Violin(x=df["churn"][df["churn"] == "No"], 
                                       y=churn_no, 
                                       name='Retenido', 
                                       line_color='#457b9d', 
                                       side='negative'))

        fig_violin.update_traces(meanline_visible=True)
        fig_violin.update_layout(template="plotly_white", showlegend=False, height=500, 
                                 yaxis_title="Cargos Mensuales ($)")
        
        st.plotly_chart(fig_violin, use_container_width=True)

        st.markdown("""
        <div class="observation-box">
        <b>Discusión de Resultados:</b> Se observa una clara bimodalidad en la distribución. Los clientes retenidos tienden a concentrarse 
        en rangos de facturación bajos (<$40), mientras que la densidad de clientes fugados aumenta drásticamente en el rango superior (>$70). 
        Esto sugiere una alta sensibilidad al precio en el segmento premium, posiblemente debido a una propuesta de valor percibida insuficiente.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Tasa de Fuga por Tipo de Contrato")
        
        # Calcular tasas
        contract_churn = df.groupby("contract")["churn"].apply(lambda x: (x=="Yes").mean()*100).reset_index()
        contract_churn = contract_churn.sort_values("churn", ascending=False)
        
        fig_bar = px.bar(contract_churn, x="churn", y="contract", orientation='h',
                         text_auto='.1f', color_discrete_sequence=['#e63946'])
        
        fig_bar.update_layout(template="plotly_white", 
                              xaxis_title="Tasa de Fuga (%)", 
                              yaxis_title="Tipo de Contrato",
                              height=400)
        
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("""
        <div class="observation-box">
        <b>Discusión de Resultados:</b> La evidencia es contundente respecto a la temporalidad contractual. El contrato 'Month-to-month' 
        presenta una volatilidad crítica, superando significativamente a los contratos de largo plazo. La ausencia de barreras de salida 
        contractuales actúa como un catalizador inmediato ante cualquier fricción en el servicio.
        </div>
        """, unsafe_allow_html=True)
        
    with tab3:
        st.markdown("### Impacto de la Fibra Óptica")
        # Matriz de calor simple o gráfico de barras agrupado
        internet_churn = df.groupby(["internet_service", "churn"]).size().reset_index(name="count")
        
        fig_col = px.bar(internet_churn, x="internet_service", y="count", color="churn",
                         color_discrete_map={"Yes": "#e63946", "No": "#457b9d"},
                         barmode="group")
        
        fig_col.update_layout(template="plotly_white", xaxis_title="Tipo de Internet", height=400)
        st.plotly_chart(fig_col, use_container_width=True)
        
        st.markdown("""
        <div class="observation-box">
        <b>Discusión de Resultados:</b> El servicio de Fibra Óptica presenta una anomalía: a pesar de ser tecnología superior, 
        concentra el mayor volumen absoluto de fugas. Esto valida la hipótesis de problemas técnicos recurrentes o una relación 
        precio-calidad desajustada en este segmento específico.
        </div>
        """, unsafe_allow_html=True)

    # --- IV. ANALÍTICA PRESCRIPTIVA ---
    st.markdown("## IV. Analítica Prescriptiva y Estrategia")

    # Matriz Lógica HTML
    st.markdown("""
    <table class="logic-table">
      <tr>
        <th>Nivel Descriptivo (Hallazgo)</th>
        <th>Nivel Diagnóstico (Causa Raíz)</th>
        <th>Nivel Prescriptivo (Acción)</th>
      </tr>
      <tr>
        <td>Alta fuga en contratos mes a mes (>40%).</td>
        <td>Bajas barreras de salida y alta sensibilidad a ofertas de competencia.</td>
        <td>Implementar <b>'Plan Migración 12M'</b>: Oferta de descuento del 15% condicionada a permanencia de 1 año.</td>
      </tr>
      <tr>
        <td>Fuga concentrada en Fibra Óptica de alto costo.</td>
        <td>Discrepancia entre precio ($70+) y valor percibido (posibles fallas técnicas).</td>
        <td>Auditoría técnica prioritaria a usuarios de Fibra + Bonificación proactiva por inactividad detectada.</td>
      </tr>
      <tr>
        <td>Clientes nuevos (<12 meses) son los más vulnerables.</td>
        <td>Falta de 'Onboarding' y adopción temprana de servicios.</td>
        <td>Campaña de <b>'Primeros 90 Días'</b>: Contacto humano educativo para asegurar uso de servicios.</td>
      </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="prescription-box">
    <b>📍 Plan de Acción Inmediato:</b><br>
    1. <b>Segmentación de Urgencia:</b> Aislar a los clientes con <i>Month-to-month</i> + <i>Fiber Optic</i>.<br>
    2. <b>Intervención Comercial:</b> Desplegar fuerza de ventas para renegociación de contratos antes del mes 12.<br>
    3. <b>Monitoreo Técnico:</b> Establecer un umbral de alerta si el uso de datos en Fibra cae un 20% (proxy de insatisfacción).
    </div>
    """, unsafe_allow_html=True)

    # --- V. CONCLUSIONES ---
    st.markdown("## V. Conclusiones Generales")
    st.markdown("""
    El análisis confirma que el precio y la flexibilidad contractual son vectores de riesgo combinados. 
    La estrategia no debe basarse en bajar precios generalizados, sino en **incentivar la migración a contratos anuales** donde el CLV (Customer Lifetime Value) se maximiza y el riesgo se diluye.
    """)
