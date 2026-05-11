"""
Simulador Educativo de Inversión en Marketing Digital
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Simulador de Inversión", page_icon="🎯", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2rem}
.section-title{font-size:.95rem;font-weight:700;color:#1e293b;border-bottom:2px solid #e2e8f0;
  padding-bottom:.3rem;margin:1.2rem 0 .7rem}
.insight-card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
  padding:.65rem 1rem;margin:.3rem 0;font-size:.87rem;color:#334155;line-height:1.5}
.info-box{background:#eff6ff;border-left:4px solid #3b82f6;border-radius:6px;
  padding:.55rem .9rem;font-size:.82rem;color:#1e40af;margin:.4rem 0}
.warn-box{background:#fffbeb;border-left:4px solid #f59e0b;border-radius:6px;
  padding:.55rem .9rem;font-size:.82rem;color:#92400e;margin:.4rem 0}
.pill-blue{display:inline-block;background:#dbeafe;color:#1e40af;border-radius:20px;
  padding:2px 11px;font-size:.78rem;font-weight:600}
.pill-purple{display:inline-block;background:#ede9fe;color:#6d28d9;border-radius:20px;
  padding:2px 11px;font-size:.78rem;font-weight:600}
.rec-box{background:linear-gradient(135deg,#f0f9ff,#f0fdf4);border-left:4px solid #3b82f6;
  border-radius:8px;padding:1rem 1.2rem;font-size:.88rem;color:#1e293b;line-height:1.7}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

CH_COLORS = {
    "Meta Ads":              "#3b82f6",
    "Google Search":         "#10b981",
    "TikTok Ads":            "#ec4899",
    "Display / Remarketing": "#f59e0b",
    "Email / CRM":           "#8b5cf6",
    "SEO / Organic":         "#06b6d4",
}
PALETTE = list(CH_COLORS.values())

CAMPAIGN_TYPES = ["Always On", "Día D", "Lanzamiento", "Retención", "Recorte"]
OBJECTIVES     = ["Conversiones", "Ingresos", "ROAS", "Utilidad bruta", "Crecimiento"]
ATT_MODELS     = ["Last Click", "First Click", "Lineal", "Position-Based", "Time Decay", "Data-Driven Simulado"]
SCENARIOS      = ["Base", "Performance", "Awareness", "Balanceado", "Recorte", "Día D"]
FUNNEL_ROLES   = ["Awareness", "Consideración", "Cierre", "Retención", "Awareness/Retención"]

SATURATION_MULT = {"Baja": 1.00, "Media": 0.85, "Alta": 0.70}

ATTRIBUTION_ROLE_W = {
    "Last Click":    {"Awareness":0.05,"Consideración":0.15,"Cierre":1.00,"Retención":0.50,"Awareness/Retención":0.30},
    "First Click":   {"Awareness":1.00,"Consideración":0.20,"Cierre":0.10,"Retención":0.20,"Awareness/Retención":0.70},
    "Lineal":        {"Awareness":1.00,"Consideración":1.00,"Cierre":1.00,"Retención":1.00,"Awareness/Retención":1.00},
    "Position-Based":{"Awareness":0.80,"Consideración":0.30,"Cierre":0.80,"Retención":0.50,"Awareness/Retención":0.65},
    "Time Decay":    {"Awareness":0.20,"Consideración":0.45,"Cierre":1.00,"Retención":0.40,"Awareness/Retención":0.30},
    "Data-Driven Simulado": None,
}

SCENARIO_ALLOC = {
    "Base":        [0.278, 0.333, 0.111, 0.089, 0.056, 0.133],
    "Performance": [0.200, 0.420, 0.080, 0.120, 0.060, 0.120],
    "Awareness":   [0.350, 0.150, 0.250, 0.050, 0.050, 0.150],
    "Balanceado":  [0.250, 0.250, 0.150, 0.100, 0.100, 0.150],
    "Recorte":     [0.200, 0.380, 0.040, 0.040, 0.140, 0.200],
    "Día D":       [0.300, 0.350, 0.200, 0.050, 0.050, 0.050],
}

SCENARIO_DESC = {
    "Base":        "Distribución de referencia sin cambios.",
    "Performance": "Concentra en Google Search y Display para maximizar conversiones directas.",
    "Awareness":   "Refuerza Meta y TikTok para descubrimiento de marca y nuevo alcance.",
    "Balanceado":  "Reparte equitativamente entre etapas del funnel.",
    "Recorte":     "Eficiencia máxima: corta canales caros, mantiene rentables.",
    "Día D":       "Pauta masiva para un evento puntual: lanzamiento o fecha clave.",
}

ATT_MODEL_DESC = {
    "Last Click":           "100% del crédito al último canal antes de la conversión. Simple, pero ignora el journey previo.",
    "First Click":          "100% del crédito al primer canal. Valora el descubrimiento de marca.",
    "Lineal":               "Crédito repartido en partes iguales entre todos los canales del journey.",
    "Position-Based":       "40% al canal de entrada + 40% al de cierre + 20% entre los intermedios.",
    "Time Decay":           "Los canales más recientes acumulan más crédito. Penaliza el awareness lejano.",
    "Data-Driven Simulado": "Crédito proporcional a adstock × confianza de tracking. Simula contribución real.",
}

OBJECTIVE_TIPS = {
    "Conversiones":   "Prioriza CPA bajo. Enfoca en canales con alta tasa de conversión y CPC razonable.",
    "Ingresos":       "Maximiza ticket × volumen. Evalúa canales con ticket promedio alto.",
    "ROAS":           "Maximiza ingresos por sol invertido. Corta canales con ROAS < 2×.",
    "Utilidad bruta": "El margen importa más que el revenue. Un buen ROAS puede tener margen bajo.",
    "Crecimiento":    "Invierte en awareness y canales con baja saturación. Piensa a mediano plazo.",
}

DEFAULT_DATA = [
    {"canal":"Meta Ads",              "inv. actual":25000,"inv. simulada":25000,"CPC":0.85,"tasa conv. (%)":2.5,"ticket promedio":180,"margen bruto (%)":42,"saturación":"Media","rol funnel":"Awareness/Retención","lag (días)":3, "adstock":1.15,"tracking (%)":70},
    {"canal":"Google Search",         "inv. actual":30000,"inv. simulada":30000,"CPC":1.20,"tasa conv. (%)":4.8,"ticket promedio":210,"margen bruto (%)":40,"saturación":"Media","rol funnel":"Cierre",             "lag (días)":1, "adstock":1.05,"tracking (%)":90},
    {"canal":"TikTok Ads",            "inv. actual":10000,"inv. simulada":10000,"CPC":0.55,"tasa conv. (%)":1.2,"ticket promedio":150,"margen bruto (%)":44,"saturación":"Baja", "rol funnel":"Awareness",           "lag (días)":5, "adstock":1.25,"tracking (%)":55},
    {"canal":"Display / Remarketing", "inv. actual":8000, "inv. simulada":8000, "CPC":0.40,"tasa conv. (%)":2.0,"ticket promedio":195,"margen bruto (%)":41,"saturación":"Alta", "rol funnel":"Consideración",       "lag (días)":2, "adstock":1.10,"tracking (%)":75},
    {"canal":"Email / CRM",           "inv. actual":5000, "inv. simulada":5000, "CPC":0.10,"tasa conv. (%)":6.5,"ticket promedio":220,"margen bruto (%)":48,"saturación":"Baja", "rol funnel":"Retención",           "lag (días)":1, "adstock":1.00,"tracking (%)":85},
    {"canal":"SEO / Organic",         "inv. actual":12000,"inv. simulada":12000,"CPC":0.25,"tasa conv. (%)":3.2,"ticket promedio":175,"margen bruto (%)":46,"saturación":"Baja", "rol funnel":"Awareness",           "lag (días)":14,"adstock":1.30,"tracking (%)":60},
]

# ═══════════════════════════════════════════════════════════════════
# CALCULATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b and not (isinstance(b, float) and np.isnan(b)) else default


def apply_scenario(df: pd.DataFrame, budget: float, scenario: str) -> pd.DataFrame:
    df = df.copy()
    ratios = SCENARIO_ALLOC.get(scenario, SCENARIO_ALLOC["Base"])
    for i in range(len(df)):
        r = ratios[i] if i < len(ratios) else 1 / len(df)
        df.at[i, "inv. simulada"] = round(budget * r)
    return df


def calc_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        inv     = max(float(r["inv. simulada"]), 0)
        cpc     = max(float(r["CPC"]), 0.001)
        tcr     = float(r["tasa conv. (%)"]) / 100
        tp      = max(float(r["ticket promedio"]), 0)
        mb      = float(r["margen bruto (%)"]) / 100
        sat_m   = SATURATION_MULT.get(str(r["saturación"]), 1.0)
        adstock = max(float(r["adstock"]), 0.01)

        clicks   = safe_div(inv, cpc)
        convs    = clicks * tcr * sat_m * adstock
        revenue  = convs * tp
        margin   = revenue * mb
        cpa      = safe_div(inv, convs)
        roas     = safe_div(revenue, inv)

        rows.append({
            "canal": r["canal"],
            "inversión": inv,
            "clicks": round(clicks),
            "conversiones": round(convs, 1),
            "ingresos": round(revenue, 2),
            "utilidad bruta": round(margin, 2),
            "CPA": round(cpa, 2),
            "ROAS": round(roas, 2),
        })
    return pd.DataFrame(rows)


def calc_base_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["inv. simulada"] = df2["inv. actual"]
    return calc_metrics(df2)


def calc_attribution(df_m: pd.DataFrame, df_in: pd.DataFrame, model: str) -> pd.DataFrame:
    if model == "Data-Driven Simulado":
        w = df_in["adstock"].values * (df_in["tracking (%)"].values / 100)
    else:
        rw = ATTRIBUTION_ROLE_W[model]
        w  = df_in["rol funnel"].map(rw).fillna(1.0).values

    raw_w   = w * df_m["conversiones"].values
    total_w = raw_w.sum()
    norm    = raw_w / total_w if total_w > 0 else np.ones(len(df_m)) / max(len(df_m), 1)

    result = df_m[["canal"]].copy()
    result["conv. atribuidas"]      = (norm * df_m["conversiones"].sum()).round(1)
    result["ingresos atribuidos"]   = (norm * df_m["ingresos"].sum()).round(2)
    result["crédito (%)"]           = (norm * 100).round(1)
    return result.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _ch_colors(channels):
    return [CH_COLORS.get(c, "#94a3b8") for c in channels]

def _base_layout(fig, title, ytitle, height=280):
    fig.update_layout(
        title_text=title, height=height,
        margin=dict(l=0, r=0, t=36, b=8),
        yaxis_title=ytitle,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.14),
    )


def chart_inversion(df: pd.DataFrame) -> go.Figure:
    chs = df["canal"].tolist()
    fig = go.Figure([
        go.Bar(name="Actual",   x=chs, y=df["inv. actual"].tolist(),  marker_color="#cbd5e1", opacity=0.85),
        go.Bar(name="Simulada", x=chs, y=df["inv. simulada"].tolist(), marker_color=_ch_colors(chs),
               text=[f"{v:,.0f}" for v in df["inv. simulada"]], textposition="outside"),
    ])
    fig.update_layout(barmode="group")
    _base_layout(fig, "Inversión por canal", "S/")
    return fig


def chart_conversiones(df_m: pd.DataFrame, df_b: pd.DataFrame) -> go.Figure:
    chs = df_m["canal"].tolist()
    fig = go.Figure([
        go.Bar(name="Base",     x=chs, y=df_b["conversiones"].tolist(), marker_color="#cbd5e1", opacity=0.85),
        go.Bar(name="Simulado", x=chs, y=df_m["conversiones"].tolist(), marker_color=_ch_colors(chs),
               text=[f"{v:.0f}" for v in df_m["conversiones"]], textposition="outside"),
    ])
    fig.update_layout(barmode="group")
    _base_layout(fig, "Conversiones estimadas", "Conversiones")
    return fig


def chart_roas(df_m: pd.DataFrame, df_b: pd.DataFrame) -> go.Figure:
    chs = df_m["canal"].tolist()
    fig = go.Figure([
        go.Bar(name="Base",     x=chs, y=df_b["ROAS"].tolist(), marker_color="#cbd5e1", opacity=0.85),
        go.Bar(name="Simulado", x=chs, y=df_m["ROAS"].tolist(), marker_color=_ch_colors(chs),
               text=[f"{v:.1f}×" for v in df_m["ROAS"]], textposition="outside"),
    ])
    fig.update_layout(barmode="group")
    fig.add_hline(y=2, line_dash="dash", line_color="#ef4444", opacity=0.5,
                  annotation_text="Mínimo 2×", annotation_position="bottom right")
    _base_layout(fig, "ROAS por canal", "ROAS")
    return fig


def chart_cpa(df_m: pd.DataFrame, df_b: pd.DataFrame) -> go.Figure:
    chs  = df_m["canal"].tolist()
    vcpa = df_m["CPA"].replace(0, np.nan)
    bcpa = df_b["CPA"].replace(0, np.nan)
    fig  = go.Figure([
        go.Bar(name="Base",     x=chs, y=bcpa.tolist(), marker_color="#cbd5e1", opacity=0.85),
        go.Bar(name="Simulado", x=chs, y=vcpa.tolist(), marker_color=_ch_colors(chs),
               text=[f"{v:,.0f}" if not np.isnan(v) else "—" for v in vcpa], textposition="outside"),
    ])
    fig.update_layout(barmode="group")
    _base_layout(fig, "CPA por canal (S/)", "S/")
    return fig


def chart_credito(df_att: pd.DataFrame, model: str) -> go.Figure:
    chs = df_att["canal"].tolist()
    fig = go.Figure(go.Bar(
        x=chs, y=df_att["crédito (%)"].tolist(),
        marker_color=_ch_colors(chs),
        text=[f"{v:.1f}%" for v in df_att["crédito (%)"]],
        textposition="outside",
    ))
    _base_layout(fig, f"Crédito atribuido — {model}", "Crédito (%)")
    fig.update_layout(yaxis_ticksuffix="%")
    return fig


def chart_ingresos_att(df_att: pd.DataFrame, df_m: pd.DataFrame) -> go.Figure:
    chs = df_att["canal"].tolist()
    fig = go.Figure([
        go.Bar(name="Generados",  x=chs, y=df_m["ingresos"].tolist(),                 marker_color="#cbd5e1", opacity=0.85),
        go.Bar(name="Atribuidos", x=chs, y=df_att["ingresos atribuidos"].tolist(), marker_color=_ch_colors(chs),
               text=[f"{v:,.0f}" for v in df_att["ingresos atribuidos"]], textposition="outside"),
    ])
    fig.update_layout(barmode="group")
    _base_layout(fig, "Ingresos atribuidos vs generados", "S/")
    return fig


# ═══════════════════════════════════════════════════════════════════
# INSIGHTS
# ═══════════════════════════════════════════════════════════════════

def generate_insights(df_m: pd.DataFrame, df_in: pd.DataFrame,
                      objective: str, att_model: str) -> list:
    ins = []

    # Mejor ROAS
    pos = df_m[df_m["ROAS"] > 0]
    if not pos.empty:
        b = pos.loc[pos["ROAS"].idxmax()]
        ins.append(("✅", f"<b>{b['canal']}</b> tiene el mejor ROAS del escenario "
                    f"(<b>{b['ROAS']:.1f}×</b>). Si su saturación es baja, escalar aquí es eficiente."))

    # Peor CPA
    pc = df_m[df_m["CPA"] > 0]
    if not pc.empty:
        w = pc.loc[pc["CPA"].idxmax()]
        ins.append(("⚠️", f"<b>{w['canal']}</b> tiene el CPA más alto "
                    f"(S/ <b>{w['CPA']:,.0f}</b>). Revisa creatividades, targeting o landing page."))

    # Saturación alta
    hi = df_in[df_in["saturación"] == "Alta"]["canal"].tolist()
    if hi:
        ins.append(("🔴", f"<b>{', '.join(hi)}</b>: saturación alta. "
                    "Cada sol adicional rinde ~30% menos que en condiciones normales."))

    # Tracking bajo
    lt = df_in[df_in["tracking (%)"] < 65]["canal"].tolist()
    if lt:
        ins.append(("🔍", f"<b>{', '.join(lt)}</b>: confianza de tracking baja (<65%). "
                    "Su impacto real podría ser mejor que lo simulado. Mejora el tagging."))

    # Last Click con sobre-inversión en cierre
    if att_model == "Last Click":
        ci = df_in[df_in["rol funnel"] == "Cierre"]["inv. simulada"].sum()
        ti = df_in["inv. simulada"].sum()
        pct = safe_div(ci, ti) * 100
        if pct > 50:
            ins.append(("⚠️", f"Con <b>Last Click</b>, el {pct:.0f}% del presupuesto está en canales de "
                        "<b>Cierre</b>. Riesgo: el modelo ignora el awareness que genera esa demanda."))

    # CRM subinvertido
    crm = df_in[df_in["canal"].str.contains("Email|CRM", case=False)]
    if not crm.empty:
        crm_pct = safe_div(crm["inv. simulada"].sum(), df_in["inv. simulada"].sum()) * 100
        crm_tcr = crm["tasa conv. (%)"].mean()
        if crm_pct < 8 and crm_tcr > 4:
            ins.append(("💡", f"<b>Email/CRM</b> recibe solo el {crm_pct:.1f}% del presupuesto "
                        f"con {crm_tcr:.1f}% de tasa de conversión. Canal subexplotado."))

    # Awareness insuficiente para objetivo de crecimiento
    if objective == "Crecimiento":
        aw_inv = df_in[df_in["rol funnel"].isin(["Awareness", "Awareness/Retención"])]["inv. simulada"].sum()
        aw_pct = safe_div(aw_inv, df_in["inv. simulada"].sum()) * 100
        if aw_pct < 25:
            ins.append(("💡", f"Solo el {aw_pct:.0f}% del presupuesto va a <b>Awareness</b>. "
                        "Para un objetivo de Crecimiento se recomienda al menos 25–35%."))

    # Canales con lag alto
    lag_hi = df_in[df_in["lag (días)"] > 7]["canal"].tolist()
    if lag_hi:
        ins.append(("⏳", f"<b>{', '.join(lag_hi)}</b>: lag > 7 días. "
                    "Su efecto en conversiones aparecerá semanas después. No los evalúes solo con métricas semanales."))

    return ins


# ═══════════════════════════════════════════════════════════════════
# RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════

def generate_recommendation(df_m: pd.DataFrame, df_in: pd.DataFrame, df_att: pd.DataFrame,
                             objective: str, att_model: str,
                             campaign_type: str, scenario: str, sym: str) -> str:
    ti    = df_in["inv. simulada"].sum()
    tc    = df_m["conversiones"].sum()
    tr    = df_m["ingresos"].sum()
    tmar  = df_m["utilidad bruta"].sum()
    roas  = safe_div(tr, ti)
    cpa   = safe_div(ti, tc)
    best  = df_m.loc[df_m["ROAS"].idxmax(), "canal"] if not df_m.empty else "—"
    top_a = df_att.loc[df_att["crédito (%)"].idxmax(), "canal"] if not df_att.empty else "—"
    hi_s  = df_in[df_in["saturación"] == "Alta"]["canal"].tolist()

    roas_txt = "sólido (≥3×)" if roas >= 3 else ("aceptable (2–3×)" if roas >= 2 else "bajo (<2×, revisar)")

    parts = [
        f"**Escenario {scenario}** · Campaña: {campaign_type} · Objetivo: {objective} · Modelo: {att_model}",
        "",
        f"Con **{sym} {ti:,.0f}** de inversión simulada, el modelo estima **{tc:,.0f} conversiones** "
        f"y **{sym} {tr:,.0f}** en ingresos. ROAS general: **{roas:.1f}×** ({roas_txt}). "
        f"CPA promedio: **{sym} {cpa:,.0f}**. Utilidad bruta estimada: **{sym} {tmar:,.0f}**.",
    ]

    obj_advice = {
        "Conversiones":   f"Concentra incrementos en **{best}** (mejor ROAS) y considera subir Email/CRM si no está saturado.",
        "Ingresos":       f"Prioriza canales con ticket alto y tasa de conversión sólida. La utilidad bruta estimada es {sym} {tmar:,.0f}.",
        "ROAS":           f"ROAS {roas:.1f}× es {roas_txt}. {'Corta canales con ROAS < 1.5× y redistribuye a ' + best + '.' if roas < 2 else 'Mantén la mezcla y optimiza creatividades en canales con saturación alta.'}",
        "Utilidad bruta": f"Prioriza canales con margen > 45%. Un ROAS alto con margen bajo puede no justificar la inversión.",
        "Crecimiento":    "Refuerza TikTok y SEO (bajo CPC, saturación baja, adstock alto). El impacto se ve en 2–4 semanas por el lag.",
    }
    parts.append("")
    parts.append(obj_advice.get(objective, ""))

    if att_model == "Last Click":
        parts.append(f"\n⚠️ **Last Click** asigna el mayor crédito a **{top_a}**, pero puede estar ignorando "
                     "el trabajo previo de canales de awareness. Valida con un modelo multi-touch antes de cortes.")
    elif att_model == "Data-Driven Simulado":
        parts.append(f"\nEl modelo **Data-Driven** pondera adstock × confianza de tracking. "
                     "Mejora el tagging de canales con tracking bajo para obtener pesos más precisos.")
    else:
        parts.append(f"\n**{att_model}** distribuye el crédito de forma más equitativa, reduciendo el riesgo "
                     "de subestimar canales de funnel medio.")

    if hi_s:
        parts.append(f"\n🔴 **{', '.join(hi_s)}** con saturación alta: el presupuesto marginal aquí rinde "
                     "~30% menos. Redirige ese excedente a canales con capacidad de escala.")

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎯 Simulador de Inversión")
    st.markdown("*Decisiones de presupuesto basadas en datos*")
    st.divider()

    sym = st.text_input("Símbolo de moneda", "S/")
    budget = st.number_input(
        "Presupuesto total", min_value=1_000, max_value=10_000_000,
        value=90_000, step=1_000, format="%d",
        help="Total disponible para distribuir entre todos los canales.",
    )

    st.divider()
    campaign_type = st.selectbox("Tipo de campaña", CAMPAIGN_TYPES)
    objective     = st.selectbox("Objetivo principal", OBJECTIVES)
    st.markdown(f'<div class="info-box">{OBJECTIVE_TIPS[objective]}</div>', unsafe_allow_html=True)

    st.divider()
    att_model = st.selectbox("Modelo de atribución", ATT_MODELS)
    st.markdown(f'<div class="info-box">{ATT_MODEL_DESC[att_model]}</div>', unsafe_allow_html=True)

    st.divider()
    scenario = st.selectbox("Escenario de inversión", SCENARIOS)
    st.markdown(f'<div class="info-box">{SCENARIO_DESC[scenario]}</div>', unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Resetear a datos de ejemplo", use_container_width=True):
        st.session_state.pop("sim_df", None)
        st.session_state.pop("_cfg", None)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE — init & scenario reset
# ═══════════════════════════════════════════════════════════════════

if "sim_df" not in st.session_state:
    st.session_state["sim_df"] = pd.DataFrame(DEFAULT_DATA)

cfg_key = f"{scenario}_{budget}"
if st.session_state.get("_cfg") != cfg_key:
    st.session_state["_cfg"] = cfg_key
    st.session_state["sim_df"] = apply_scenario(st.session_state["sim_df"], budget, scenario)


# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown("## 📊 Simulador de Inversión en Marketing Digital")
hc1, hc2 = st.columns([3, 1])
with hc1:
    st.markdown(
        f'<span class="pill-blue">Escenario: {scenario}</span>&nbsp;&nbsp;'
        f'<span class="pill-purple">Modelo: {att_model}</span>',
        unsafe_allow_html=True,
    )
with hc2:
    st.markdown(
        f'<div style="text-align:right;font-size:.82rem;color:#64748b;padding-top:4px">'
        f'Campaña: <b>{campaign_type}</b> &nbsp;|&nbsp; Objetivo: <b>{objective}</b></div>',
        unsafe_allow_html=True,
    )
st.divider()


# ═══════════════════════════════════════════════════════════════════
# DATA EDITOR
# ═══════════════════════════════════════════════════════════════════

st.markdown('<p class="section-title">📋 Parámetros por canal — edita para simular</p>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="info-box">Modifica <b>inv. simulada</b>, <b>CPC</b>, <b>tasa de conversión</b>, '
    '<b>saturación</b> o cualquier otro parámetro. Los KPIs y gráficos se actualizan al instante. '
    'El canal y la <b>inv. actual</b> son de referencia (no editables).</div>',
    unsafe_allow_html=True,
)

col_cfg = {
    "canal": st.column_config.TextColumn("Canal", disabled=True, width="medium"),
    "inv. actual": st.column_config.NumberColumn(
        f"Inv. actual ({sym})", disabled=True, format=f"{sym} %,.0f",
        help="Inversión de referencia (no editable)."),
    "inv. simulada": st.column_config.NumberColumn(
        f"Inv. simulada ({sym})", min_value=0, format=f"{sym} %,.0f",
        help="Modifica este valor para simular el nuevo escenario."),
    "CPC": st.column_config.NumberColumn(
        f"CPC ({sym})", min_value=0.01, max_value=50.0, format=f"{sym} %.2f",
        help="Costo por clic = inversión ÷ clicks."),
    "tasa conv. (%)": st.column_config.NumberColumn(
        "Conv. (%)", min_value=0.0, max_value=100.0, step=0.1, format="%.1f %%",
        help="% de clics que se convierten en pedidos o leads."),
    "ticket promedio": st.column_config.NumberColumn(
        f"Ticket ({sym})", min_value=0, format=f"{sym} %,.0f",
        help="Valor promedio por conversión."),
    "margen bruto (%)": st.column_config.NumberColumn(
        "Margen (%)", min_value=0, max_value=100, step=1, format="%.0f %%",
        help="% de los ingresos que queda como utilidad bruta después de COGS."),
    "saturación": st.column_config.SelectboxColumn(
        "Saturación", options=["Baja", "Media", "Alta"],
        help="Baja: sin penalización. Media: −15% eficiencia. Alta: −30% eficiencia."),
    "rol funnel": st.column_config.SelectboxColumn(
        "Rol en el funnel", options=FUNNEL_ROLES,
        help="Determina el peso del canal según el modelo de atribución seleccionado."),
    "lag (días)": st.column_config.NumberColumn(
        "Lag (días)", min_value=0, max_value=60,
        help="Días de retraso promedio entre inversión y conversión observada."),
    "adstock": st.column_config.NumberColumn(
        "Adstock", min_value=0.5, max_value=2.5, step=0.05, format="%.2f",
        help="Multiplicador de carryover: >1 indica que la pauta acumula efecto en el tiempo."),
    "tracking (%)": st.column_config.NumberColumn(
        "Tracking (%)", min_value=0, max_value=100, step=5, format="%.0f %%",
        help="Confianza en la medición de conversiones. Bajo = datos incompletos."),
}

edited = st.data_editor(
    st.session_state["sim_df"],
    column_config=col_cfg,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    key="editor_channels",
)
st.session_state["sim_df"] = edited


# ═══════════════════════════════════════════════════════════════════
# COMPUTE
# ═══════════════════════════════════════════════════════════════════

df_m  = calc_metrics(edited)
df_b  = calc_base_metrics(edited)
df_at = calc_attribution(df_m, edited, att_model)

ti   = float(edited["inv. simulada"].sum())
ti_b = float(edited["inv. actual"].sum())
tc   = df_m["conversiones"].sum()
tc_b = df_b["conversiones"].sum()
tr   = df_m["ingresos"].sum()
tr_b = df_b["ingresos"].sum()
tmar = df_m["utilidad bruta"].sum()
cpa  = safe_div(ti, tc)
cpa_b= safe_div(ti_b, tc_b)
roas = safe_div(tr, ti)
roas_b = safe_div(tr_b, ti_b)


# ═══════════════════════════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════════════════════════

st.divider()
st.markdown('<p class="section-title">📌 KPIs del escenario simulado</p>', unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric(f"Inversión total ({sym})", f"{ti:,.0f}",    delta=f"{ti-ti_b:+,.0f}")
k2.metric("Conversiones",             f"{tc:,.0f}",    delta=f"{tc-tc_b:+,.0f}")
k3.metric(f"Ingresos ({sym})",        f"{tr:,.0f}",    delta=f"{tr-tr_b:+,.0f}")
k4.metric(f"Utilidad bruta ({sym})",  f"{tmar:,.0f}",  help="Ingresos × margen bruto ponderado.")
k5.metric(f"CPA ({sym})",             f"{cpa:,.0f}",   delta=f"{cpa-cpa_b:+,.0f}", delta_color="inverse")
k6.metric("ROAS total",               f"{roas:.2f}×",  delta=f"{roas-roas_b:+.2f}×")


# ═══════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════

st.divider()
st.markdown('<p class="section-title">📊 Visualizaciones comparativas (base vs simulado)</p>',
            unsafe_allow_html=True)

r1c1, r1c2 = st.columns(2)
with r1c1: st.plotly_chart(chart_inversion(edited),            use_container_width=True)
with r1c2: st.plotly_chart(chart_conversiones(df_m, df_b),    use_container_width=True)

r2c1, r2c2 = st.columns(2)
with r2c1: st.plotly_chart(chart_roas(df_m, df_b),            use_container_width=True)
with r2c2: st.plotly_chart(chart_cpa(df_m, df_b),             use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# ATTRIBUTION
# ═══════════════════════════════════════════════════════════════════

st.divider()
st.markdown(f'<p class="section-title">🏅 Atribución de conversiones — {att_model}</p>',
            unsafe_allow_html=True)

a_left, a_right = st.columns([3, 2])
with a_left:
    ac1, ac2 = st.columns(2)
    with ac1: st.plotly_chart(chart_credito(df_at, att_model),        use_container_width=True)
    with ac2: st.plotly_chart(chart_ingresos_att(df_at, df_m),        use_container_width=True)

with a_right:
    st.markdown(
        f'<div class="info-box"><b>¿Qué muestra esta sección?</b><br><br>'
        f'{ATT_MODEL_DESC[att_model]}<br><br>'
        f'El mismo revenue puede distribuirse de forma muy diferente según el modelo. '
        f'Cambiar el modelo en el sidebar para ver cómo varía el crédito por canal.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="section-title">Tabla de crédito</p>', unsafe_allow_html=True)
    att_show = df_at.rename(columns={
        "canal": "Canal",
        "crédito (%)": "Crédito (%)",
        "conv. atribuidas": "Conv. atribuidas",
        "ingresos atribuidos": f"Ingresos atribuidos ({sym})",
    })
    st.dataframe(att_show, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════

st.divider()
st.markdown('<p class="section-title">📄 Resultados simulados por canal</p>', unsafe_allow_html=True)

res = df_m.rename(columns={
    "canal": "Canal",
    "inversión": f"Inversión ({sym})",
    "clicks": "Clicks",
    "conversiones": "Conversiones",
    "ingresos": f"Ingresos ({sym})",
    "utilidad bruta": f"Utilidad bruta ({sym})",
    "CPA": f"CPA ({sym})",
    "ROAS": "ROAS",
})
st.dataframe(
    res.style.format({
        f"Inversión ({sym})":     "{:,.0f}",
        "Clicks":                 "{:,.0f}",
        "Conversiones":           "{:.1f}",
        f"Ingresos ({sym})":      "{:,.0f}",
        f"Utilidad bruta ({sym})":"{:,.0f}",
        f"CPA ({sym})":           "{:,.0f}",
        "ROAS":                   "{:.2f}×",
    }),
    use_container_width=True,
    hide_index=True,
)


# ═══════════════════════════════════════════════════════════════════
# INSIGHTS
# ═══════════════════════════════════════════════════════════════════

st.divider()
st.markdown('<p class="section-title">💡 Insights automáticos</p>', unsafe_allow_html=True)

insights = generate_insights(df_m, edited, objective, att_model)
if insights:
    for icon, text in insights:
        st.markdown(f'<div class="insight-card">{icon}&nbsp; {text}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="insight-card">✅ No se detectaron alertas en este escenario.</div>',
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════

st.divider()
st.markdown('<p class="section-title">📝 Recomendación estratégica</p>', unsafe_allow_html=True)

rec = generate_recommendation(df_m, edited, df_at, objective, att_model, campaign_type, scenario, sym)

with st.container():
    st.markdown('<div class="rec-box">', unsafe_allow_html=True)
    st.markdown(rec)
    st.markdown('</div>', unsafe_allow_html=True)
