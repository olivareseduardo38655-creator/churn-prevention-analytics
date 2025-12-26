# Sistema de Analítica Predictiva y Retención de Clientes (Churn)

![Python](https://img.shields.io/badge/Language-Python_3.10-00599C?style=flat-square&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit_Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/XAI-SHAP-555555?style=flat-square)
![Power BI](https://img.shields.io/badge/BI-Power_BI-F2C811?style=flat-square&logo=powerbi&logoColor=black)
![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=flat-square)

## Descripción Ejecutiva
Este proyecto implementa un sistema integral de **Inteligencia Artificial Explicable (XAI)** diseñado para mitigar la tasa de cancelación de clientes (Churn) en el sector de telecomunicaciones.

El sistema ingesta datos transaccionales y demográficos para entrenar un modelo de aprendizaje supervisado (Random Forest). A diferencia de los modelos de "caja negra" tradicionales, esta solución integra valores **SHAP (SHapley Additive exPlanations)** para determinar la causalidad individual de cada predicción. El resultado final es un dataset enriquecido ("Gold Data") que alimenta un tablero de control en Power BI, permitiendo a los equipos de negocio segmentar riesgos y ejecutar estrategias de retención basadas en evidencia de datos.

## Objetivos Técnicos
1.  **Arquitectura Modular:** Implementación de principios de *Clean Architecture* para desacoplar la ingesta, el procesamiento y el modelado.
2.  **Explicabilidad del Modelo:** Integración de bibliotecas XAI para transformar probabilidades matemáticas en razones de negocio legibles (ej. "Riesgo alto debido a contrato mensual").
3.  **Ingeniería de Características:** Transformación de variables crudas en cohortes de comportamiento (`tenure_group`, `service_density`).
4.  **Integración BI:** Generación automatizada de una capa de datos analítica optimizada para la ingesta directa en Microsoft Power BI.

## Arquitectura del Sistema

```mermaid
graph LR
    subgraph Data Engineering
    A[Raw Data Simulation] -->|Generation| B[Data Processing]
    B -->|Cleaning & Encoding| C[ML Input Dataset]
    end

    subgraph Data Science Core
    C --> D{Random Forest Model}
    D -->|Inference| E[Churn Probabilities]
    D -->|SHAP Analysis| F[Feature Importance & Reason Codes]
    end

    subgraph Business Intelligence
    E & F --> G[Gold Dataset Transformation]
    G --> H((Power BI Dashboard))
    end
