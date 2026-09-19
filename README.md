# UPC · Modelos de Atribución de Marketing Digital

Suite de apps en Streamlit para enseñar e ilustrar modelos de atribución de
marketing (multi-touch attribution) y simulación de inversión en medios,
desarrolladas para el curso de Digital Marketing Analytics (UPC).

## Apps incluidas

### 1. `attribution_app.py` — Attribution Analyzer

Analiza customer journeys (rutas de canales) y compara modelos de atribución
sobre datos reales o cargados por el usuario.

- **Análisis descriptivo**: journeys, tasa de conversión, distribución de touchpoints.
- **Modelos de atribución**: First Click, Last Click, Last Non-Direct, Lineal,
  Time Decay, U-Shape, Markov Chain, Data-Driven.
- **Escenarios de inversión**: comparación de reasignación de presupuesto por canal.
- **Guías integradas**: cómo preparar el dataset y cómo interpretar resultados.

Dataset de ejemplo: `attribution_dataset.csv` (journey_id, path, num_touchpoints,
converted, revenue).

### 2. `simulador_inversion.py` — Simulador de Inversión en Marketing Digital

Simulador educativo para explorar el efecto de distintos modelos de atribución
y escenarios de inversión sobre canales típicos (Meta Ads, Google Search,
TikTok Ads, Display/Remarketing, Email/CRM, SEO/Organic), tipos de campaña,
objetivos de negocio y saturación de medios.

## Requisitos

- Python 3.12+
- Ver `requirements.txt` para dependencias exactas (Streamlit, Pandas, NumPy, Plotly, etc.)

## Instalación

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
# Attribution Analyzer
streamlit run attribution_app.py

# Simulador de Inversión
streamlit run simulador_inversion.py
```

Cada app se abre en `http://localhost:8501` por defecto.

## Estructura del proyecto

```
upc-atribucion/
├── attribution_app.py         # App: análisis y modelos de atribución
├── simulador_inversion.py     # App: simulador de inversión en medios
├── attribution_dataset.csv    # Dataset de ejemplo (journeys sintéticos)
└── requirements.txt           # Dependencias del proyecto
```

## Notas

- Los datos incluidos son sintéticos, pensados para fines didácticos.
- No se debe usar información real de clientes en las demos del curso.
