import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
from collections import Counter

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Attribution Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .section-title {
        font-size: 1.05rem; font-weight: 600; color: #1e293b;
        margin-bottom: 0.5rem; padding-bottom: 0.35rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .info-box {
        background: #eff6ff; border-left: 4px solid #3b82f6;
        border-radius: 6px; padding: 0.7rem 1rem;
        font-size: 0.86rem; color: #1e40af; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#3b82f6","#10b981","#f59e0b","#ec4899","#8b5cf6","#06b6d4","#ef4444","#84cc16"]
MODEL_COLORS = {
    "First Click": "#3b82f6",
    "Last Click":  "#10b981",
    "Lineal":      "#f59e0b",
    "Time Decay":  "#ec4899",
    "Markov":      "#8b5cf6",
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def parse_path(p):
    return [c.strip() for c in str(p).split(">") if c.strip()]

def first_click(conv):
    d = {}
    for _, r in conv.iterrows():
        p = parse_path(r["path"]); v = float(r["revenue"])
        if p: d[p[0]] = d.get(p[0], 0) + v
    return d

def last_click(conv):
    d = {}
    for _, r in conv.iterrows():
        p = parse_path(r["path"]); v = float(r["revenue"])
        if p: d[p[-1]] = d.get(p[-1], 0) + v
    return d

def linear_model(conv):
    d = {}
    for _, r in conv.iterrows():
        p = parse_path(r["path"]); v = float(r["revenue"]); n = len(p)
        if n: [d.__setitem__(c, d.get(c,0)+v/n) for c in p]
    return d

def time_decay(conv, decay=0.7):
    d = {}
    for _, r in conv.iterrows():
        p = parse_path(r["path"]); v = float(r["revenue"]); n = len(p)
        if not n: continue
        w = [np.exp(decay*(i-n+1)) for i in range(n)]; ws = sum(w)
        for i,c in enumerate(p): d[c] = d.get(c,0) + v*w[i]/ws
    return d

def markov_model(df, channels):
    states = ["__start__"] + channels + ["__conv__","__null__"]
    tr = {s:{t:0 for t in states} for s in states}
    for _, r in df.iterrows():
        p = parse_path(r["path"])
        full = ["__start__"] + p + [("__conv__" if int(r["converted"])==1 else "__null__")]
        for i in range(len(full)-1):
            f,t = full[i], full[i+1]
            if f in tr and t in tr: tr[f][t] += 1
    norm = {}
    for f in states:
        tot = sum(tr[f].values())
        norm[f] = {t: tr[f][t]/tot if tot>0 else 0 for t in states}
    def rate(excluded=None):
        act = [s for s in states if s!=excluded]
        t2 = {s:{e:norm[s].get(e,0) for e in act} for s in act}
        for s in act:
            if s in ("__conv__","__null__"): continue
            rs = sum(t2[s].values())
            if rs>0: t2[s] = {e:v/rs for e,v in t2[s].items()}
        v = {s:(1.0 if s=="__conv__" else 0.0) for s in act}
        for _ in range(300):
            nv={s:v[s] for s in act}
            for s in act:
                if s in ("__conv__","__null__"): continue
                nv[s]=sum(t2[s].get(e,0)*v.get(e,0) for e in act)
            v=nv
        return v.get("__start__",0)
    base = rate()
    eff = {c:max(0,base-rate(c)) for c in channels}
    te = sum(eff.values())
    total_rev = float(df[df["converted"]==1]["revenue"].sum())
    return {c:(eff[c]/te*total_rev if te>0 else 0) for c in channels}

def compute_models(df, channels):
    conv = df[df["converted"]==1].copy()
    res = {
        "First Click": first_click(conv),
        "Last Click":  last_click(conv),
        "Lineal":      linear_model(conv),
        "Time Decay":  time_decay(conv),
        "Markov":      markov_model(df, channels),
    }
    for m in res:
        for c in channels: res[m].setdefault(c,0)
    return res

def to_pct(d, channels):
    tot = sum(d.get(c,0) for c in channels)
    return {c:(d.get(c,0)/tot*100 if tot>0 else 0) for c in channels}

# ─────────────────────────────────────────────
# EXCEL EXPORT
# ─────────────────────────────────────────────
def build_excel(sheets_dict):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book
        title_fmt  = wb.add_format({"bold":True,"font_size":13,"font_color":"#1e293b","bottom":2,"border_color":"#3b82f6"})
        hdr_fmt    = wb.add_format({"bold":True,"bg_color":"#1e40af","font_color":"#ffffff","border":1,"border_color":"#93c5fd","align":"center"})
        fmts = {
            "pct":   wb.add_format({"num_format":'0.0"%"',"border":1,"border_color":"#e2e8f0"}),
            "money": wb.add_format({"num_format":"#,##0.00","border":1,"border_color":"#e2e8f0"}),
            "int":   wb.add_format({"num_format":"#,##0","border":1,"border_color":"#e2e8f0"}),
            "text":  wb.add_format({"border":1,"border_color":"#e2e8f0"}),
        }
        alt_fmts = {
            "pct":   wb.add_format({"bg_color":"#eff6ff","num_format":'0.0"%"',"border":1,"border_color":"#e2e8f0"}),
            "money": wb.add_format({"bg_color":"#eff6ff","num_format":"#,##0.00","border":1,"border_color":"#e2e8f0"}),
            "int":   wb.add_format({"bg_color":"#eff6ff","num_format":"#,##0","border":1,"border_color":"#e2e8f0"}),
            "text":  wb.add_format({"bg_color":"#eff6ff","border":1,"border_color":"#e2e8f0"}),
        }
        for sname, (df_s, col_fmts, title) in sheets_dict.items():
            df_s.to_excel(writer, sheet_name=sname[:31], index=False, startrow=2)
            ws = writer.sheets[sname[:31]]
            ws.write(0, 0, title, title_fmt)
            for ci, cn in enumerate(df_s.columns): ws.write(2, ci, cn, hdr_fmt)
            for ri in range(len(df_s)):
                alt = ri%2==1
                for ci, cn in enumerate(df_s.columns):
                    fk = col_fmts.get(cn,"text")
                    ws.write(ri+3, ci, df_s.iloc[ri,ci], alt_fmts[fk] if alt else fmts[fk])
            for ci, cn in enumerate(df_s.columns):
                ml = max(len(str(cn)), df_s[cn].astype(str).str.len().max())
                ws.set_column(ci, ci, min(ml+4, 35))
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Attribution Analyzer")
    st.markdown("*Modelos de atribución multicanal*")
    st.divider()
    st.markdown("### Fuente de datos")
    data_src = st.radio("", ["📁 Cargar CSV propio","🧪 Dataset de ejemplo"], label_visibility="collapsed")

    df_raw = None
    if data_src == "📁 Cargar CSV propio":
        upl = st.file_uploader("Sube tu CSV", type=["csv"])
        if upl:
            df_raw = pd.read_csv(upl)
            st.success(f"✓ {len(df_raw):,} registros cargados")
    else:
        import os
        sp = "/mnt/user-data/outputs/attribution_dataset.csv"
        if os.path.exists(sp):
            df_raw = pd.read_csv(sp)
            st.success(f"✓ Dataset de ejemplo ({len(df_raw):,} journeys)")
        else:
            st.warning("Dataset no encontrado. Sube un CSV.")

    df = None
    channels = []

    if df_raw is not None:
        st.divider()
        st.markdown("### Mapeo de columnas")
        cols = list(df_raw.columns)
        col_path = st.selectbox("Path del journey", cols, index=cols.index("path") if "path" in cols else 0)
        col_conv = st.selectbox("Convertido (0/1)", cols, index=cols.index("converted") if "converted" in cols else 0)
        col_rev  = st.selectbox("Revenue", cols, index=cols.index("revenue") if "revenue" in cols else 0)
        curr_sym = st.text_input("Símbolo moneda", "S/")

        df = df_raw.rename(columns={col_path:"path", col_conv:"converted", col_rev:"revenue"})
        df["converted"] = pd.to_numeric(df["converted"], errors="coerce").fillna(0).astype(int)
        df["revenue"]   = pd.to_numeric(df["revenue"],   errors="coerce").fillna(0)

        all_raw = []
        for p in df["path"]: all_raw.extend(parse_path(p))
        channels = sorted(set(all_raw))

        st.divider()
        st.markdown("### Presupuesto")
        budget = st.number_input("Total a invertir", min_value=1000, value=100000, step=1000, format="%d")
    else:
        curr_sym = "S/"
        budget = 100000

# ─────────────────────────────────────────────
# WELCOME SCREEN
# ─────────────────────────────────────────────
if df is None:
    st.markdown("## 👋 Attribution Analyzer")
    st.markdown("""
    <div class="info-box">
    Carga tu dataset de customer journeys o usa el dataset de ejemplo desde la barra lateral.
    </div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Formato del CSV**")
        st.dataframe(pd.DataFrame({
            "journey_id":[1,2,3],
            "path":["Organic Search > Email","Paid Search","Social Media > Direct"],
            "converted":[1,0,1],
            "revenue":[350.0,0,820.0],
        }), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Modelos disponibles**")
        for m,color in MODEL_COLORS.items():
            st.markdown(f"<span style='display:inline-block;width:12px;height:12px;background:{color};border-radius:3px;margin-right:6px;vertical-align:middle;'></span> **{m}**", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────────
conv_df   = df[df["converted"]==1]
total_rev = float(conv_df["revenue"].sum())
n_conv    = int(df["converted"].sum())
n_total   = len(df)
conv_rate = n_conv/n_total*100
avg_order = total_rev/n_conv if n_conv>0 else 0

all_raw_channels = []
for p in df["path"]: all_raw_channels.extend(parse_path(p))

df["n_tp"] = df["path"].apply(lambda p: len(parse_path(p)))
df["first_ch"] = df["path"].apply(lambda p: parse_path(p)[0] if parse_path(p) else None)
df["last_ch"]  = df["path"].apply(lambda p: parse_path(p)[-1] if parse_path(p) else None)

with st.spinner("Calculando modelos de atribución (Markov puede tardar unos segundos)..."):
    results  = compute_models(df, channels)
pct_res  = {m: to_pct(results[m], channels) for m in results}
model_names = list(results.keys())

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈  Análisis descriptivo", "🏅  Modelos de atribución", "🎯  Escenarios de inversión"])

# ══════════════════════════════════════════════
# TAB 1 — DESCRIPTIVE
# ══════════════════════════════════════════════
with tab1:
    # KPIs
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Journeys",       f"{n_total:,}")
    k2.metric("Conversiones",   f"{n_conv:,}")
    k3.metric("Tasa conv.",      f"{conv_rate:.1f}%")
    k4.metric(f"Revenue ({curr_sym})", f"{total_rev:,.0f}")
    k5.metric(f"Ticket prom. ({curr_sym})", f"{avg_order:,.0f}")
    st.divider()

    # Row 1
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-title">Frecuencia de touchpoints por canal</p>', unsafe_allow_html=True)
        ch_cnt = Counter(all_raw_channels)
        ch_df_plot = pd.DataFrame({"Canal":list(ch_cnt.keys()),"N":list(ch_cnt.values())}).sort_values("N")
        fig1 = go.Figure(go.Bar(
            x=ch_df_plot["N"], y=ch_df_plot["Canal"], orientation="h",
            marker_color=[PALETTE[i%len(PALETTE)] for i in range(len(ch_df_plot))],
            text=ch_df_plot["N"], textposition="outside",
        ))
        fig1.update_layout(height=280, margin=dict(l=0,r=50,t=10,b=10),
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.markdown('<p class="section-title">Distribución de longitud de journey (touchpoints)</p>', unsafe_allow_html=True)
        tp_vc = df["n_tp"].value_counts().sort_index().reset_index()
        tp_vc.columns = ["Touchpoints","Journeys"]
        fig2 = go.Figure(go.Bar(
            x=tp_vc["Touchpoints"], y=tp_vc["Journeys"],
            marker_color="#3b82f6", text=tp_vc["Journeys"], textposition="outside",
        ))
        fig2.update_layout(height=280, margin=dict(l=0,r=10,t=10,b=10),
                           xaxis_title="Nro. touchpoints", yaxis_title="Journeys",
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<p class="section-title">Tasa de conversión: primer vs. último canal</p>', unsafe_allow_html=True)
        fc_conv = df.groupby("first_ch")["converted"].mean()*100
        lc_conv = df.groupby("last_ch")["converted"].mean()*100
        all_ch2 = sorted(set(fc_conv.index)|set(lc_conv.index))
        fig3 = go.Figure([
            go.Bar(name="Primer canal", x=all_ch2, y=[round(fc_conv.get(c,0),1) for c in all_ch2], marker_color="#3b82f6"),
            go.Bar(name="Último canal", x=all_ch2, y=[round(lc_conv.get(c,0),1) for c in all_ch2], marker_color="#10b981"),
        ])
        fig3.update_layout(barmode="group", height=280, margin=dict(l=0,r=0,t=30,b=10),
                           yaxis_title="Conv. rate (%)", plot_bgcolor="white", paper_bgcolor="white",
                           legend=dict(orientation="h",y=1.12))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<p class="section-title">Distribución del revenue (journeys convertidos)</p>', unsafe_allow_html=True)
        fig4 = go.Figure(go.Histogram(x=conv_df["revenue"], nbinsx=30, marker_color="#8b5cf6", opacity=0.8))
        fig4.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=10),
                           xaxis_title=f"Revenue ({curr_sym})", yaxis_title="Conversiones",
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig4, use_container_width=True)

    # Row 3: Revenue por canal (last click proxy)
    st.markdown('<p class="section-title">Revenue promedio de conversión por canal (último touchpoint)</p>', unsafe_allow_html=True)
    rev_ch = conv_df.groupby("last_ch")["revenue"].agg(["mean","sum","count"]).reset_index()
    rev_ch.columns = ["Canal","Ticket promedio","Revenue total","Conversiones"]
    fig5 = go.Figure([
        go.Bar(name="Ticket promedio", x=rev_ch["Canal"], y=rev_ch["Ticket promedio"].round(0),
               marker_color="#3b82f6", yaxis="y1"),
        go.Scatter(name="Revenue total", x=rev_ch["Canal"], y=rev_ch["Revenue total"].round(0),
                   mode="lines+markers", line=dict(color="#ec4899",width=2), marker=dict(size=8), yaxis="y2"),
    ])
    fig5.update_layout(
        height=300, margin=dict(l=0,r=60,t=30,b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title=f"Ticket prom. ({curr_sym})"),
        yaxis2=dict(title=f"Revenue total ({curr_sym})", overlaying="y", side="right"),
        legend=dict(orientation="h",y=1.12),
    )
    st.plotly_chart(fig5, use_container_width=True)

    # Top paths
    st.markdown('<p class="section-title">Top 10 journeys más frecuentes</p>', unsafe_allow_html=True)
    path_stats = df.groupby("path").agg(
        Journeys=("converted","count"),
        Conversiones=("converted","sum"),
        Revenue=("revenue","sum")
    ).reset_index()
    path_stats["Conv. Rate (%)"] = (path_stats["Conversiones"]/path_stats["Journeys"]*100).round(1)
    path_stats["Revenue"] = path_stats["Revenue"].round(0)
    top_paths = path_stats.sort_values("Journeys",ascending=False).head(10)
    st.dataframe(top_paths.rename(columns={"path":"Journey path"}), use_container_width=True, hide_index=True)

    # Export
    st.divider()
    st.markdown('<p class="section-title">📤 Exportar análisis descriptivo</p>', unsafe_allow_html=True)
    exp1, _ = st.columns([1,3])
    with exp1:
        kpi_df = pd.DataFrame({
            "Métrica":["Journeys totales","Conversiones","Tasa de conversión (%)","Revenue total","Ticket promedio"],
            "Valor":[n_total, n_conv, round(conv_rate,1), round(total_rev,0), round(avg_order,0)]
        })
        ch_freq_df = pd.DataFrame({"Canal":list(ch_cnt.keys()),"Touchpoints":list(ch_cnt.values())}).sort_values("Touchpoints",ascending=False)
        cr_df = pd.DataFrame({"Canal":all_ch2,"Conv% primer canal":[round(fc_conv.get(c,0),1) for c in all_ch2],
                               "Conv% último canal":[round(lc_conv.get(c,0),1) for c in all_ch2]})
        rev_ch2 = rev_ch.copy()
        rev_ch2["Ticket promedio"] = rev_ch2["Ticket promedio"].round(0)
        rev_ch2["Revenue total"]   = rev_ch2["Revenue total"].round(0)
        tp_exp = tp_vc.copy()
        sheets_d = {
            "KPIs":              (kpi_df, {"Métrica":"text","Valor":"text"}, "Resumen del dataset"),
            "Frecuencia canales":(ch_freq_df, {"Canal":"text","Touchpoints":"int"}, "Touchpoints por canal"),
            "Conv rate canal":   (cr_df, {"Canal":"text","Conv% primer canal":"pct","Conv% último canal":"pct"}, "Tasas de conversión"),
            "Revenue por canal": (rev_ch2, {"Canal":"text","Ticket promedio":"money","Revenue total":"money","Conversiones":"int"}, "Revenue por canal (último click)"),
            "Long. journey":     (tp_exp, {"Touchpoints":"int","Journeys":"int"}, "Distribución de journeys"),
            "Top journeys":      (top_paths.rename(columns={"path":"Journey path"}),
                                  {"Journey path":"text","Journeys":"int","Conversiones":"int","Revenue":"money","Conv. Rate (%)":"pct"}, "Top 10 journeys"),
        }
        buf1 = build_excel(sheets_d)
        st.download_button("⬇️ Descargar descriptivo (.xlsx)", buf1, "descriptivo.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══════════════════════════════════════════════
# TAB 2 — ATTRIBUTION MODELS
# ══════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="info-box">
    Créditos calculados sobre el revenue de journeys convertidos.
    Markov (data-driven) mide cuánto cae la tasa de conversión al eliminar cada canal del grafo de transiciones.
    </div>
    """, unsafe_allow_html=True)

    # Grouped bar
    st.markdown('<p class="section-title">Crédito atribuido por canal y modelo (%)</p>', unsafe_allow_html=True)
    fig_comp = go.Figure()
    for m in model_names:
        fig_comp.add_trace(go.Bar(
            name=m, x=channels,
            y=[round(pct_res[m][c],1) for c in channels],
            marker_color=MODEL_COLORS[m],
            text=[f"{pct_res[m][c]:.1f}%" for c in channels],
            textposition="outside",
        ))
    fig_comp.update_layout(barmode="group", height=380,
                           yaxis_title="Crédito (%)", yaxis=dict(ticksuffix="%"),
                           plot_bgcolor="white", paper_bgcolor="white",
                           legend=dict(orientation="h",y=1.1),
                           margin=dict(l=0,r=0,t=40,b=10))
    st.plotly_chart(fig_comp, use_container_width=True)

    # Heatmap
    st.markdown('<p class="section-title">Mapa de calor — % de crédito por modelo y canal</p>', unsafe_allow_html=True)
    z = [[round(pct_res[m][c],1) for c in channels] for m in model_names]
    fig_heat = go.Figure(go.Heatmap(
        z=z, x=channels, y=model_names,
        colorscale="Blues",
        text=[[f"{v:.1f}%" for v in row] for row in z],
        texttemplate="%{text}", showscale=True,
        colorbar=dict(title="%"),
    ))
    fig_heat.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=10),
                           plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_heat, use_container_width=True)

    # Radar chart
    st.markdown('<p class="section-title">Radar — perfil de atribución por modelo</p>', unsafe_allow_html=True)
    fig_radar = go.Figure()
    for m in model_names:
        vals = [round(pct_res[m][c],1) for c in channels]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=channels+[channels[0]],
            name=m, line=dict(color=MODEL_COLORS[m], width=2),
            fill="toself", opacity=0.15,
        ))
    fig_radar.update_layout(height=380, margin=dict(l=20,r=20,t=30,b=20),
                            polar=dict(radialaxis=dict(visible=True, ticksuffix="%")),
                            legend=dict(orientation="h", y=-0.15),
                            paper_bgcolor="white")
    st.plotly_chart(fig_radar, use_container_width=True)

    # Comparison table
    st.markdown('<p class="section-title">Tabla comparativa de crédito (%)</p>', unsafe_allow_html=True)
    rows_pct = [{"Canal":c, **{m:round(pct_res[m][c],1) for m in model_names}} for c in channels]
    rows_rev = [{"Canal":c, **{f"{m} ({curr_sym})":round(results[m][c],0) for m in model_names}} for c in channels]
    comp_pct = pd.DataFrame(rows_pct)
    comp_rev = pd.DataFrame(rows_rev)
    st.dataframe(comp_pct, use_container_width=True, hide_index=True)
    st.markdown('<p class="section-title">Revenue atribuido por canal y modelo</p>', unsafe_allow_html=True)
    st.dataframe(comp_rev, use_container_width=True, hide_index=True)

    # Model guide
    st.divider()
    st.markdown('<p class="section-title">¿Cuándo usar cada modelo?</p>', unsafe_allow_html=True)
    model_info = [
        ("First Click","#3b82f6","Discovery y awareness. Premia el primer contacto.","Ignora completamente el cierre de la venta."),
        ("Last Click","#10b981","Retargeting y campañas de cierre. Simple y auditable.","Subestima canales de awareness y nurturing."),
        ("Lineal","#f59e0b","Base neutral cuando no hay hipótesis sobre qué canal importa más.","No diferencia impacto real entre canales."),
        ("Time Decay","#ec4899","Ciclos de compra cortos. Premia touchpoints cercanos a la conversión.","Penaliza canales de discovery en journeys largos."),
        ("Markov","#8b5cf6","E-commerce multicanal con journeys de 2+ pasos. Data-driven.","Requiere volumen de datos suficiente para ser estable."),
    ]
    cols_m = st.columns(5)
    for col, (m, color, pro, con) in zip(cols_m, model_info):
        with col:
            st.markdown(f"<span style='display:inline-block;width:10px;height:10px;background:{color};border-radius:50%;margin-right:5px;'></span>**{m}**", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size:0.8rem;color:#15803d;'>✔ {pro}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size:0.8rem;color:#b91c1c;'>✖ {con}</span>", unsafe_allow_html=True)

    # Export
    st.divider()
    exp2, _ = st.columns([1,3])
    with exp2:
        pct_fmts = {"Canal":"text", **{m:"pct" for m in model_names}}
        rev_fmts = {"Canal":"text", **{f"{m} ({curr_sym})":"money" for m in model_names}}
        sheets2 = {
            "Crédito (%)":       (comp_pct, pct_fmts, "Crédito por canal — todos los modelos (%)"),
            f"Crédito ({curr_sym})": (comp_rev, rev_fmts, f"Revenue atribuido ({curr_sym}) — todos los modelos"),
        }
        buf2 = build_excel(sheets2)
        st.download_button("⬇️ Descargar modelos de atribución (.xlsx)", buf2, "atribucion.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══════════════════════════════════════════════
# TAB 3 — SCENARIOS
# ══════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="info-box">
    Selecciona canales activos, elige el modelo base y ajusta los pesos para construir tu escenario de inversión.
    El presupuesto total se configura en la barra lateral.
    </div>
    """, unsafe_allow_html=True)

    sc1, sc2 = st.columns([1,2])

    with sc1:
        scenario_name = st.text_input("Nombre del escenario", "Escenario A")
        base_model    = st.selectbox("Modelo base", model_names, index=4)
        active_chs    = st.multiselect("Canales activos", channels, default=channels)
        if not active_chs:
            st.warning("Selecciona al menos un canal.")
            st.stop()

        base_pcts = pct_res[base_model]
        base_active_tot = sum(base_pcts.get(c,0) for c in active_chs)
        base_norm = {c: (base_pcts.get(c,0)/base_active_tot*100 if base_active_tot>0 else 0) for c in active_chs}

        st.markdown("**Ajuste manual de pesos (%)**")
        weights = {}
        for c in active_chs:
            weights[c] = st.slider(c, 0.0, 100.0, float(round(base_norm.get(c,0),1)), step=0.5, format="%.1f%%")

        total_w = sum(weights.values())
        norm = {c:(weights[c]/total_w*100 if total_w>0 else 0) for c in active_chs}
        alloc = {c: norm[c]/100*budget for c in active_chs}

    with sc2:
        # Comparison: base model vs scenario
        fig_sc1 = go.Figure([
            go.Bar(name=f"Modelo base ({base_model})", x=active_chs,
                   y=[round(base_norm[c],1) for c in active_chs],
                   marker_color="#94a3b8", opacity=0.7,
                   text=[f"{base_norm[c]:.1f}%" for c in active_chs], textposition="outside"),
            go.Bar(name=f"{scenario_name}", x=active_chs,
                   y=[round(norm[c],1) for c in active_chs],
                   marker_color="#3b82f6",
                   text=[f"{norm[c]:.1f}%" for c in active_chs], textposition="outside"),
        ])
        fig_sc1.update_layout(barmode="group", height=300,
                              yaxis_title="% del presupuesto", yaxis=dict(ticksuffix="%"),
                              plot_bgcolor="white", paper_bgcolor="white",
                              legend=dict(orientation="h",y=1.12),
                              margin=dict(l=0,r=0,t=40,b=10))
        st.plotly_chart(fig_sc1, use_container_width=True)

        # Allocation donut
        fig_donut = go.Figure(go.Pie(
            labels=active_chs,
            values=[round(alloc[c],0) for c in active_chs],
            hole=0.55,
            marker_colors=[PALETTE[i%len(PALETTE)] for i in range(len(active_chs))],
            textinfo="label+percent",
            hovertemplate="%{label}<br>%{value:,.0f} "+curr_sym+"<extra></extra>",
        ))
        fig_donut.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=10),
                                paper_bgcolor="white",
                                annotations=[dict(text=f"{curr_sym}<br>{budget:,.0f}", x=0.5, y=0.5,
                                                  font_size=13, showarrow=False)])
        st.plotly_chart(fig_donut, use_container_width=True)

        # Table
        bud_rows = []
        for c in active_chs:
            delta = round(norm[c]-base_norm[c],1)
            bud_rows.append({
                "Canal": c,
                f"Base {base_model} (%)": round(base_norm[c],1),
                "Escenario (%)": round(norm[c],1),
                f"Inversión ({curr_sym})": round(alloc[c],0),
                "Δ (pp)": f"{'+'if delta>=0 else ''}{delta}",
            })
        bud_df = pd.DataFrame(bud_rows)
        st.dataframe(bud_df, use_container_width=True, hide_index=True)

    # Multi-model scenario comparison
    st.divider()
    st.markdown('<p class="section-title">Comparar presupuesto según cada modelo (sin ajuste manual)</p>', unsafe_allow_html=True)
    multi_rows = []
    for c in channels:
        row = {"Canal": c}
        for m in model_names:
            p_active = {ch: pct_res[m].get(ch,0) for ch in channels}
            tot = sum(p_active.values())
            row[f"{m} ({curr_sym})"] = round(p_active[c]/tot*budget if tot>0 else 0, 0)
        multi_rows.append(row)
    multi_df = pd.DataFrame(multi_rows)

    fig_multi = go.Figure()
    for m in model_names:
        fig_multi.add_trace(go.Bar(
            name=m, x=multi_df["Canal"], y=multi_df[f"{m} ({curr_sym})"],
            marker_color=MODEL_COLORS[m],
            text=[f"{curr_sym} {int(v):,}" for v in multi_df[f"{m} ({curr_sym})"]],
            textposition="outside",
        ))
    fig_multi.update_layout(barmode="group", height=350,
                            yaxis_title=f"Inversión ({curr_sym})",
                            plot_bgcolor="white", paper_bgcolor="white",
                            legend=dict(orientation="h",y=1.1),
                            margin=dict(l=0,r=0,t=40,b=10))
    st.plotly_chart(fig_multi, use_container_width=True)
    st.dataframe(multi_df, use_container_width=True, hide_index=True)

    # Export
    st.divider()
    exp3a, exp3b, exp3c = st.columns([1,1,2])
    with exp3a:
        bud_exp = bud_df.copy()
        bud_exp[f"Base {base_model} (%)"] = bud_exp[f"Base {base_model} (%)"].astype(float)
        bud_exp["Escenario (%)"]          = bud_exp["Escenario (%)"].astype(float)
        bud_exp[f"Inversión ({curr_sym})"]= bud_exp[f"Inversión ({curr_sym})"].astype(float)
        bud_fmts = {"Canal":"text", f"Base {base_model} (%)":"pct", "Escenario (%)":"pct",
                    f"Inversión ({curr_sym})":"money", "Δ (pp)":"text"}
        sheets3 = {
            "Escenario activo": (bud_exp, bud_fmts, f"Escenario: {scenario_name}"),
            "Multi-modelo":     (multi_df, {"Canal":"text",**{f"{m} ({curr_sym})":"money" for m in model_names}},
                                 f"Inversión {curr_sym} {budget:,.0f} por modelo"),
        }
        buf3 = build_excel(sheets3)
        st.download_button("⬇️ Descargar escenarios (.xlsx)", buf3, "escenarios.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with exp3b:
        # Full report
        all_sheets = {
            "KPIs":               (pd.DataFrame({"Métrica":["Journeys","Conversiones","Tasa conv. (%)","Revenue total","Ticket prom."],
                                                  "Valor":[n_total,n_conv,round(conv_rate,1),round(total_rev,0),round(avg_order,0)]}),
                                   {"Métrica":"text","Valor":"text"}, "Resumen del dataset"),
            "Frec. canales":      (pd.DataFrame({"Canal":list(ch_cnt.keys()),"Touchpoints":list(ch_cnt.values())}).sort_values("Touchpoints",ascending=False),
                                   {"Canal":"text","Touchpoints":"int"}, "Touchpoints por canal"),
            "Conv rate canal":    (cr_df, {"Canal":"text","Conv% primer canal":"pct","Conv% último canal":"pct"}, "Tasas de conversión"),
            "Top journeys":       (top_paths.rename(columns={"path":"Journey path"}),
                                   {"Journey path":"text","Journeys":"int","Conversiones":"int","Revenue":"money","Conv. Rate (%)":"pct"}, "Top 10 journeys"),
            "Crédito (%)":        (comp_pct, pct_fmts, "Crédito por canal — todos los modelos (%)"),
            f"Crédito rev.":      (comp_rev, rev_fmts, f"Revenue atribuido ({curr_sym})"),
            "Escenario activo":   (bud_exp, bud_fmts, f"Escenario: {scenario_name}"),
            "Multi-modelo":       (multi_df, {"Canal":"text",**{f"{m} ({curr_sym})":"money" for m in model_names}},
                                   f"Inversión por modelo"),
        }
        buf_all = build_excel(all_sheets)
        st.download_button("⬇️ Reporte completo (.xlsx)", buf_all, "attribution_reporte_completo.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           type="primary")
