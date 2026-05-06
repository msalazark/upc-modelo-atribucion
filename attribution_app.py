import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import math
from itertools import chain, combinations
from collections import Counter

st.set_page_config(page_title="Attribution Analyzer", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:4rem;padding-bottom:2rem}
.section-title{font-size:1rem;font-weight:600;color:#1e293b;margin:1rem 0 0.4rem;
  padding-bottom:0.3rem;border-bottom:2px solid #e2e8f0}
.info-box{background:#eff6ff;border-left:4px solid #3b82f6;border-radius:6px;
  padding:.6rem 1rem;font-size:.85rem;color:#1e40af;margin-bottom:.8rem}
.warn-box{background:#fffbeb;border-left:4px solid #f59e0b;border-radius:6px;
  padding:.6rem 1rem;font-size:.85rem;color:#92400e;margin-bottom:.8rem}
.ok-box{background:#f0fdf4;border-left:4px solid #22c55e;border-radius:6px;
  padding:.6rem 1rem;font-size:.85rem;color:#15803d;margin-bottom:.8rem}
.model-card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;
  padding:.9rem 1rem;margin-bottom:.6rem}
.guide-box{background:#fafafa;border:1px solid #e2e8f0;border-radius:8px;
  padding:1rem 1.2rem;margin-bottom:1rem}
</style>
""", unsafe_allow_html=True)

PALETTE = ["#3b82f6","#10b981","#f59e0b","#ec4899","#8b5cf6","#06b6d4","#ef4444","#84cc16"]
MODEL_COLORS = {
    "First Click":     "#3b82f6",
    "Last Click":      "#10b981",
    "Last Non-Direct": "#06b6d4",
    "Lineal":          "#f59e0b",
    "Time Decay":      "#ec4899",
    "U-Shape":         "#06b6d4",
    "Markov":          "#8b5cf6",
    "Data-Driven":     "#ef4444",
}
CH_TYPE_COLOR = {"paid":"#3b82f6","seo":"#10b981","direct":"#94a3b8"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_path(p):
    return [c.strip() for c in str(p).split(">") if c.strip()]

def first_click(conv):
    d={}
    for _,r in conv.iterrows():
        p=parse_path(r["path"]); v=float(r["revenue"])
        if p: d[p[0]]=d.get(p[0],0)+v
    return d

def last_click(conv):
    d={}
    for _,r in conv.iterrows():
        p=parse_path(r["path"]); v=float(r["revenue"])
        if p: d[p[-1]]=d.get(p[-1],0)+v
    return d

def last_non_direct(conv, skip_chs):
    d={}
    for _,r in conv.iterrows():
        p=parse_path(r["path"]); v=float(r["revenue"])
        eff=[c for c in p if c not in skip_chs]
        ch=eff[-1] if eff else (p[-1] if p else None)
        if ch: d[ch]=d.get(ch,0)+v
    return d

def linear_model(conv):
    d={}
    for _,r in conv.iterrows():
        p=parse_path(r["path"]); v=float(r["revenue"]); n=len(p)
        if n:
            for c in p: d[c]=d.get(c,0)+v/n
    return d

def time_decay(conv, decay=0.7):
    d={}
    for _,r in conv.iterrows():
        p=parse_path(r["path"]); v=float(r["revenue"]); n=len(p)
        if not n: continue
        w=[np.exp(decay*(i-n+1)) for i in range(n)]; ws=sum(w)
        for i,c in enumerate(p): d[c]=d.get(c,0)+v*w[i]/ws
    return d

def u_shape_model(conv):
    d={}
    for _,r in conv.iterrows():
        p=parse_path(r["path"]); v=float(r["revenue"]); n=len(p)
        if n==0: continue
        elif n==1: d[p[0]]=d.get(p[0],0)+v
        elif n==2:
            d[p[0]]=d.get(p[0],0)+v*0.5
            d[p[1]]=d.get(p[1],0)+v*0.5
        else:
            d[p[0]]=d.get(p[0],0)+v*0.4
            d[p[-1]]=d.get(p[-1],0)+v*0.4
            mid=p[1:-1]; share=v*0.2/len(mid)
            for c in mid: d[c]=d.get(c,0)+share
    return d

def data_driven_shapley(df, channels):
    channels=list(channels)
    n=len(channels)
    ch_set=set(channels)
    coalition_conv={}
    coalition_count={}
    for _,row in df.iterrows():
        path_chs=tuple(sorted(set(parse_path(row["path"])) & ch_set))
        conv=int(row["converted"])
        coalition_conv[path_chs]=coalition_conv.get(path_chs,0)+conv
        coalition_count[path_chs]=coalition_count.get(path_chs,0)+1
    def v(coalition):
        key=tuple(sorted(coalition))
        cnt=coalition_count.get(key,0)
        if cnt==0:
            sub_rates=[
                coalition_conv.get(tuple(sorted(s)),0)/max(coalition_count.get(tuple(sorted(s)),1),1)
                for sz in range(1,len(key))
                for s in combinations(key,sz)
            ]
            return float(np.mean(sub_rates)) if sub_rates else 0.0
        return coalition_conv[key]/cnt
    shapley={}
    for c in channels:
        sv=0.0
        others=[ch for ch in channels if ch!=c]
        for subset in chain.from_iterable(combinations(others,sz) for sz in range(len(others)+1)):
            s=len(subset)
            w=(math.factorial(s)*math.factorial(n-s-1))/math.factorial(n)
            sv+=w*(v(list(subset)+[c])-v(list(subset)))
        shapley[c]=max(sv,0.0)
    total_rev=float(df[df["converted"]==1]["revenue"].sum())
    tot=sum(shapley.values())
    return {c:(shapley[c]/tot*total_rev if tot>0 else 0) for c in channels}

def markov_model(df, channels):
    states=["__start__"]+list(channels)+["__conv__","__null__"]
    tr={s:{t:0 for t in states} for s in states}
    for _,r in df.iterrows():
        p=parse_path(r["path"])
        conv=int(r["converted"])==1
        full=["__start__"]+p+[("__conv__" if conv else "__null__")]
        for i in range(len(full)-1):
            f,t=full[i],full[i+1]
            if f in tr and t in tr: tr[f][t]+=1
    norm={}
    for f in states:
        tot=sum(tr[f].values())
        norm[f]={t:(tr[f][t]/tot if tot>0 else 0) for t in states}
    def rate(excluded=None):
        act=[s for s in states if s!=excluded]
        t2={s:{e:norm[s].get(e,0) for e in act} for s in act}
        for s in act:
            if s in("__conv__","__null__"): continue
            rs=sum(t2[s].values())
            if rs>0: t2[s]={e:vv/rs for e,vv in t2[s].items()}
        vd={s:(1.0 if s=="__conv__" else 0.0) for s in act}
        for _ in range(300):
            nv={s:vd[s] for s in act}
            for s in act:
                if s in("__conv__","__null__"): continue
                nv[s]=sum(t2[s].get(e,0)*vd.get(e,0) for e in act)
            vd=nv
        return vd.get("__start__",0)
    base=rate()
    eff={c:max(0,base-rate(c)) for c in channels}
    te=sum(eff.values())
    total_rev=float(df[df["converted"]==1]["revenue"].sum())
    return {c:(eff[c]/te*total_rev if te>0 else 0) for c in channels}

def to_pct(d, channels):
    tot=sum(d.get(c,0) for c in channels)
    return {c:(d.get(c,0)/tot*100 if tot>0 else 0) for c in channels}

def compute_simple_models(df, channels, direct_chs=None):
    conv=df[df["converted"]==1].copy()
    skip=set(direct_chs or [])
    res={
        "First Click":     first_click(conv),
        "Last Click":      last_click(conv),
        "Last Non-Direct": last_non_direct(conv,skip),
        "Lineal":          linear_model(conv),
        "Time Decay":      time_decay(conv),
        "U-Shape":         u_shape_model(conv),
    }
    for m in res:
        for c in channels: res[m].setdefault(c,0)
    return res

# ── Excel export ──────────────────────────────────────────────────────────────
def build_excel(sheets_dict):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="xlsxwriter") as writer:
        wb=writer.book
        title_fmt=wb.add_format({"bold":True,"font_size":13,"font_color":"#1e293b","bottom":2,"border_color":"#3b82f6"})
        hdr=wb.add_format({"bold":True,"bg_color":"#1e40af","font_color":"#ffffff","border":1,"align":"center"})
        base={"pct":wb.add_format({"num_format":'0.0"%"',"border":1,"border_color":"#e2e8f0"}),
              "money":wb.add_format({"num_format":"#,##0.00","border":1,"border_color":"#e2e8f0"}),
              "int":wb.add_format({"num_format":"#,##0","border":1,"border_color":"#e2e8f0"}),
              "text":wb.add_format({"border":1,"border_color":"#e2e8f0"})}
        alt={"pct":wb.add_format({"bg_color":"#eff6ff","num_format":'0.0"%"',"border":1,"border_color":"#e2e8f0"}),
             "money":wb.add_format({"bg_color":"#eff6ff","num_format":"#,##0.00","border":1,"border_color":"#e2e8f0"}),
             "int":wb.add_format({"bg_color":"#eff6ff","num_format":"#,##0","border":1,"border_color":"#e2e8f0"}),
             "text":wb.add_format({"bg_color":"#eff6ff","border":1,"border_color":"#e2e8f0"})}
        for sname,(df_s,col_fmts,title) in sheets_dict.items():
            df_s.to_excel(writer,sheet_name=sname[:31],index=False,startrow=2)
            ws=writer.sheets[sname[:31]]
            ws.write(0,0,title,title_fmt)
            for ci,cn in enumerate(df_s.columns): ws.write(2,ci,cn,hdr)
            for ri in range(len(df_s)):
                a=ri%2==1
                for ci,cn in enumerate(df_s.columns):
                    fk=col_fmts.get(cn,"text")
                    ws.write(ri+3,ci,df_s.iloc[ri,ci],alt[fk] if a else base[fk])
            for ci,cn in enumerate(df_s.columns):
                ml=max(len(str(cn)),df_s[cn].astype(str).str.len().max())
                ws.set_column(ci,ci,min(ml+4,35))
    buf.seek(0)
    return buf

# Defaults (overwritten inside sidebar if data loaded)
direct_chs=[]; organic_chs=[]; seo_budget=0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Attribution Analyzer")
    st.markdown("*Modelos de atribución multicanal*")
    st.divider()

    st.markdown("### Fuente de datos")
    data_src=st.radio("",["📁 Cargar CSV propio","🧪 Dataset de ejemplo"],label_visibility="collapsed")

    df_raw=None
    if data_src=="📁 Cargar CSV propio":
        upl=st.file_uploader("Sube tu CSV",type=["csv"])
        if upl:
            try:
                df_raw=pd.read_csv(upl)
                st.success(f"✓ {len(df_raw):,} registros cargados")
            except Exception as e:
                st.error(f"Error al leer CSV: {e}")
    else:
        import os
        sp="/mnt/user-data/outputs/attribution_dataset.csv"
        if os.path.exists(sp):
            df_raw=pd.read_csv(sp)
            st.success(f"✓ Dataset de ejemplo ({len(df_raw):,} journeys)")
        else:
            st.warning("Dataset de ejemplo no encontrado.")

    col_path="path"; col_conv="converted"; col_rev="revenue"; curr_sym="S/"; budget=100000

    if df_raw is not None:
        st.divider()
        st.markdown("### Mapeo de columnas")
        cols=list(df_raw.columns)
        def pick(label,default):
            return st.selectbox(label,cols,index=cols.index(default) if default in cols else 0)
        col_path=pick("Path del journey","path")
        col_conv=pick("Convertido (0/1)","converted")
        col_rev =pick("Revenue","revenue")
        curr_sym=st.text_input("Símbolo moneda","S/")
        st.divider()
        st.markdown("### Presupuesto")
        budget=st.number_input("Presupuesto publicitario (paid)",min_value=0,value=100000,step=1000,format="%d")
        seo_budget=st.number_input("Presupuesto SEO / Contenido",min_value=0,value=20000,step=1000,format="%d",
            help="Inversión en posicionamiento orgánico y contenido — no es pauta directa")

        _raw_chs=sorted(set(c for p in df_raw[col_path].astype(str) for c in parse_path(p)))
        if _raw_chs:
            st.divider()
            st.markdown("### Tipo de canal")
            _auto_direct=[c for c in _raw_chs if any(k in c.lower() for k in ["direct","directo","brand"])]
            _auto_seo   =[c for c in _raw_chs if any(k in c.lower() for k in ["organic","orgánico","organico","seo"])]
            direct_chs =st.multiselect("🔴 Directo / No invertible",_raw_chs,default=_auto_direct,
                help="Canales que no se pueden activar con presupuesto (ej. tráfico directo)")
            organic_chs=st.multiselect("🟡 SEO / Orgánico",_raw_chs,
                default=[c for c in _auto_seo if c not in direct_chs],
                help="Invertible en análisis y contenido SEO, no en pauta")
            st.caption("El resto se considera canal **pagado**.")

# ── Welcome ───────────────────────────────────────────────────────────────────
if df_raw is None:
    st.markdown("## 👋 Attribution Analyzer")
    st.markdown('<div class="info-box">Carga tu CSV o usa el dataset de ejemplo desde la barra lateral.</div>',
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"journey_id":[1,2],"path":["Organic Search > Email","Paid Search"],
                               "converted":[1,0],"revenue":[350.0,0]}),hide_index=True)
    st.stop()

# ── Prepare ───────────────────────────────────────────────────────────────────
try:
    df=df_raw.rename(columns={col_path:"path",col_conv:"converted",col_rev:"revenue"})
    df["converted"]=pd.to_numeric(df["converted"],errors="coerce").fillna(0).astype(int)
    df["revenue"]  =pd.to_numeric(df["revenue"],  errors="coerce").fillna(0)
    df["path"]     =df["path"].astype(str)

    all_raw=[]
    for p in df["path"]: all_raw.extend(parse_path(p))
    channels=sorted(set(all_raw))
    if not channels:
        st.error("No se detectaron canales. Verifica que la columna 'path' use ' > ' como separador.")
        st.stop()

    df["n_tp"]    =df["path"].apply(lambda p:len(parse_path(p)))
    df["first_ch"]=df["path"].apply(lambda p:parse_path(p)[0] if parse_path(p) else None)
    df["last_ch"] =df["path"].apply(lambda p:parse_path(p)[-1] if parse_path(p) else None)

    conv_df  =df[df["converted"]==1].copy()
    total_rev=float(conv_df["revenue"].sum())
    n_conv   =int(df["converted"].sum())
    n_total  =len(df)
    conv_rate=n_conv/n_total*100 if n_total>0 else 0
    avg_order=total_rev/n_conv if n_conv>0 else 0

    ch_cnt=Counter(all_raw)

except Exception as e:
    st.error(f"Error procesando el dataset: {e}")
    st.exception(e)
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5=st.tabs([
    "📈  Análisis descriptivo",
    "🏅  Modelos de atribución",
    "🎯  Escenarios de inversión",
    "📋  Guía: preparar datos",
    "📖  Guía: interpretar resultados",
])

# ═══════════════════════════════════════════
# TAB 1
# ═══════════════════════════════════════════
with tab1:
    try:
        k1,k2,k3,k4,k5=st.columns(5)
        k1.metric("Journeys",f"{n_total:,}")
        k2.metric("Conversiones",f"{n_conv:,}")
        k3.metric("Tasa de conv.",f"{conv_rate:.1f}%")
        k4.metric(f"Revenue ({curr_sym})",f"{total_rev:,.0f}")
        k5.metric(f"Ticket prom. ({curr_sym})",f"{avg_order:,.0f}")
        st.divider()

        c1,c2=st.columns(2)
        with c1:
            st.markdown('<p class="section-title">Frecuencia de touchpoints por canal</p>',unsafe_allow_html=True)
            ch_df_sorted=pd.DataFrame({"Canal":list(ch_cnt.keys()),"N":list(ch_cnt.values())}).sort_values("N")
            fig1=go.Figure(go.Bar(x=ch_df_sorted["N"],y=ch_df_sorted["Canal"],orientation="h",
                marker_color=[PALETTE[i%len(PALETTE)] for i in range(len(ch_df_sorted))],
                text=ch_df_sorted["N"],textposition="outside"))
            fig1.update_layout(height=260,margin=dict(l=0,r=50,t=8,b=8),
                               plot_bgcolor="white",paper_bgcolor="white",
                               xaxis_title="Nro. de apariciones")
            st.plotly_chart(fig1,use_container_width=True)

        with c2:
            st.markdown('<p class="section-title">Distribución de longitud de journey</p>',unsafe_allow_html=True)
            tp_vc=df["n_tp"].value_counts().sort_index().reset_index()
            tp_vc.columns=["Touchpoints","Journeys"]
            fig2=go.Figure(go.Bar(x=tp_vc["Touchpoints"],y=tp_vc["Journeys"],
                marker_color="#3b82f6",text=tp_vc["Journeys"],textposition="outside"))
            fig2.update_layout(height=260,margin=dict(l=0,r=10,t=8,b=8),
                               xaxis_title="Nro. de touchpoints",yaxis_title="Journeys",
                               plot_bgcolor="white",paper_bgcolor="white")
            st.plotly_chart(fig2,use_container_width=True)

        c3,c4=st.columns(2)
        with c3:
            st.markdown('<p class="section-title">Tasa de conversión — primer vs. último canal</p>',unsafe_allow_html=True)
            fc_conv=df.groupby("first_ch")["converted"].mean()*100
            lc_conv=df.groupby("last_ch")["converted"].mean()*100
            all_ch2=sorted(set(list(fc_conv.index)+list(lc_conv.index)))
            fig3=go.Figure([
                go.Bar(name="Primer canal",x=all_ch2,y=[round(fc_conv.get(c,0),1) for c in all_ch2],marker_color="#3b82f6"),
                go.Bar(name="Último canal", x=all_ch2,y=[round(lc_conv.get(c,0),1) for c in all_ch2],marker_color="#10b981"),
            ])
            fig3.update_layout(barmode="group",height=260,margin=dict(l=0,r=0,t=30,b=8),
                               yaxis_title="Conv. rate (%)",plot_bgcolor="white",paper_bgcolor="white",
                               legend=dict(orientation="h",y=1.12))
            st.plotly_chart(fig3,use_container_width=True)

        with c4:
            st.markdown('<p class="section-title">Distribución del revenue (conversiones)</p>',unsafe_allow_html=True)
            fig4=go.Figure(go.Histogram(x=conv_df["revenue"],nbinsx=30,marker_color="#8b5cf6",opacity=0.8))
            fig4.update_layout(height=260,margin=dict(l=0,r=0,t=8,b=8),
                               xaxis_title=f"Revenue ({curr_sym})",yaxis_title="Conversiones",
                               plot_bgcolor="white",paper_bgcolor="white")
            st.plotly_chart(fig4,use_container_width=True)

        st.markdown('<p class="section-title">Revenue promedio y total por último canal (proxy Last Click)</p>',unsafe_allow_html=True)
        rev_ch=conv_df.groupby("last_ch")["revenue"].agg(["mean","sum","count"]).reset_index()
        rev_ch.columns=["Canal","Ticket promedio","Revenue total","Conversiones"]
        fig5=go.Figure([
            go.Bar(name="Ticket prom.",x=rev_ch["Canal"],y=rev_ch["Ticket promedio"].round(0),
                   marker_color="#3b82f6",yaxis="y1"),
            go.Scatter(name="Revenue total",x=rev_ch["Canal"],y=rev_ch["Revenue total"].round(0),
                       mode="lines+markers",line=dict(color="#ec4899",width=2),marker=dict(size=8),yaxis="y2"),
        ])
        fig5.update_layout(height=280,margin=dict(l=0,r=60,t=30,b=8),
                           yaxis=dict(title=f"Ticket prom. ({curr_sym})"),
                           yaxis2=dict(title=f"Revenue total ({curr_sym})",overlaying="y",side="right"),
                           plot_bgcolor="white",paper_bgcolor="white",
                           legend=dict(orientation="h",y=1.12))
        st.plotly_chart(fig5,use_container_width=True)

        st.markdown('<p class="section-title">Top 10 journeys más frecuentes</p>',unsafe_allow_html=True)
        path_stats=df.groupby("path").agg(Journeys=("converted","count"),
            Conversiones=("converted","sum"),Revenue=("revenue","sum")).reset_index()
        path_stats["Conv. Rate (%)"]=( path_stats["Conversiones"]/path_stats["Journeys"]*100).round(1)
        path_stats["Revenue"]=path_stats["Revenue"].round(0)
        top_paths=path_stats.sort_values("Journeys",ascending=False).head(10).rename(columns={"path":"Journey path"})
        st.dataframe(top_paths,use_container_width=True,hide_index=True)

        st.divider()
        st.markdown('<p class="section-title">📤 Exportar análisis descriptivo</p>',unsafe_allow_html=True)
        exp1,_=st.columns([1,3])
        with exp1:
            kpi_df=pd.DataFrame({"Métrica":["Journeys","Conversiones","Tasa conv. (%)","Revenue total","Ticket prom."],
                                  "Valor":[n_total,n_conv,round(conv_rate,1),round(total_rev,0),round(avg_order,0)]})
            ch_freq_df=pd.DataFrame({"Canal":list(ch_cnt.keys()),"Touchpoints":list(ch_cnt.values())}).sort_values("Touchpoints",ascending=False)
            cr_df=pd.DataFrame({"Canal":all_ch2,
                "Conv% primer canal":[round(fc_conv.get(c,0),1) for c in all_ch2],
                "Conv% último canal":[round(lc_conv.get(c,0),1) for c in all_ch2]})
            rev_ch_exp=rev_ch.copy()
            rev_ch_exp["Ticket promedio"]=rev_ch_exp["Ticket promedio"].round(0)
            rev_ch_exp["Revenue total"]  =rev_ch_exp["Revenue total"].round(0)
            sheets1={
                "KPIs":              (kpi_df,{"Métrica":"text","Valor":"text"},"Resumen del dataset"),
                "Frecuencia canales":(ch_freq_df,{"Canal":"text","Touchpoints":"int"},"Touchpoints por canal"),
                "Conv rate canal":   (cr_df,{"Canal":"text","Conv% primer canal":"pct","Conv% último canal":"pct"},"Tasas de conversión"),
                "Revenue por canal": (rev_ch_exp,{"Canal":"text","Ticket promedio":"money","Revenue total":"money","Conversiones":"int"},"Revenue por canal"),
                "Long. journey":     (tp_vc,{"Touchpoints":"int","Journeys":"int"},"Distribución de longitud"),
                "Top journeys":      (top_paths,{"Journey path":"text","Journeys":"int","Conversiones":"int","Revenue":"money","Conv. Rate (%)":"pct"},"Top 10 journeys"),
            }
            buf1=build_excel(sheets1)
            st.download_button("⬇️ Descargar descriptivo (.xlsx)",buf1,"descriptivo.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"Error en análisis descriptivo: {e}")
        st.exception(e)

# ═══════════════════════════════════════════
# TAB 2
# ═══════════════════════════════════════════
with tab2:
    try:
        st.markdown('<div class="info-box">Los modelos de regla (First Click, Last Click, Lineal, Time Decay, U-Shape) se calculan al instante. <b>Markov</b> y <b>Data-Driven (Shapley)</b> pueden tardar 15-30 segundos — actívalos con los botones.</div>',
                    unsafe_allow_html=True)

        simple_res=compute_simple_models(df,channels,direct_chs)
        simple_pct={m:to_pct(simple_res[m],channels) for m in simple_res}

        if "markov_result" not in st.session_state:
            st.session_state["markov_result"]=None
        if "shapley_result" not in st.session_state:
            st.session_state["shapley_result"]=None

        btn_col1,btn_col2=st.columns(2)
        with btn_col1:
            run_markov=st.button("🔮 Calcular Markov",type="secondary",use_container_width=True)
        with btn_col2:
            run_shapley=st.button("🧮 Calcular Data-Driven (Shapley)",type="primary",use_container_width=True)

        if run_markov:
            with st.spinner("Calculando cadenas de Markov... (puede tardar 15-30 segundos)"):
                try:
                    mk=markov_model(df,channels)
                    st.session_state["markov_result"]=mk
                    st.success("✓ Markov calculado correctamente")
                except Exception as e:
                    st.error(f"Error en Markov: {e}")

        if run_shapley:
            with st.spinner("Calculando Shapley values... (puede tardar 15-30 segundos según número de canales)"):
                try:
                    sh=data_driven_shapley(df,channels)
                    st.session_state["shapley_result"]=sh
                    st.success("✓ Data-Driven (Shapley) calculado correctamente")
                except Exception as e:
                    st.error(f"Error en Data-Driven (Shapley): {e}")

        all_res=dict(simple_res)
        all_pct=dict(simple_pct)
        if st.session_state["markov_result"] is not None:
            mk=st.session_state["markov_result"]
            for c in channels: mk.setdefault(c,0)
            all_res["Markov"]=mk
            all_pct["Markov"]=to_pct(mk,channels)
        if st.session_state["shapley_result"] is not None:
            sh=st.session_state["shapley_result"]
            for c in channels: sh.setdefault(c,0)
            all_res["Data-Driven"]=sh
            all_pct["Data-Driven"]=to_pct(sh,channels)
        model_names_avail=list(all_res.keys())

        st.markdown('<p class="section-title">Crédito atribuido por canal y modelo (%)</p>',unsafe_allow_html=True)
        fig_comp=go.Figure()
        for m in model_names_avail:
            fig_comp.add_trace(go.Bar(
                name=m, x=channels,
                y=[round(all_pct[m][c],1) for c in channels],
                marker_color=MODEL_COLORS.get(m,PALETTE[0]),
                text=[f"{all_pct[m][c]:.1f}%" for c in channels],
                textposition="outside",
            ))
        fig_comp.update_layout(barmode="group",height=380,
            yaxis=dict(title="Crédito (%)",ticksuffix="%"),
            plot_bgcolor="white",paper_bgcolor="white",
            legend=dict(orientation="h",y=1.1),margin=dict(l=0,r=0,t=40,b=8))
        st.plotly_chart(fig_comp,use_container_width=True)

        st.markdown('<p class="section-title">Mapa de calor — % de crédito</p>',unsafe_allow_html=True)
        z=[[round(all_pct[m][c],1) for c in channels] for m in model_names_avail]
        fig_heat=go.Figure(go.Heatmap(z=z,x=channels,y=model_names_avail,
            colorscale="Blues",text=[[f"{vv:.1f}%" for vv in row] for row in z],
            texttemplate="%{text}",showscale=True,colorbar=dict(title="%")))
        fig_heat.update_layout(height=max(220,len(model_names_avail)*50),
            margin=dict(l=0,r=0,t=8,b=8),plot_bgcolor="white",paper_bgcolor="white")
        st.plotly_chart(fig_heat,use_container_width=True)

        if len(model_names_avail)>=2:
            st.markdown('<p class="section-title">Radar — perfil de atribución por modelo</p>',unsafe_allow_html=True)
            fig_radar=go.Figure()
            for m in model_names_avail:
                vals=[round(all_pct[m][c],1) for c in channels]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals+[vals[0]],theta=channels+[channels[0]],name=m,
                    line=dict(color=MODEL_COLORS.get(m,"#888"),width=2),fill="toself",opacity=0.15))
            fig_radar.update_layout(height=380,margin=dict(l=20,r=20,t=30,b=20),
                polar=dict(radialaxis=dict(visible=True,ticksuffix="%")),
                legend=dict(orientation="h",y=-0.15),paper_bgcolor="white")
            st.plotly_chart(fig_radar,use_container_width=True)

        st.markdown('<p class="section-title">Tabla comparativa — crédito (%)</p>',unsafe_allow_html=True)
        rows_pct=[{"Canal":c,**{m:round(all_pct[m][c],1) for m in model_names_avail}} for c in channels]
        rows_rev=[{"Canal":c,**{f"{m} ({curr_sym})":round(all_res[m].get(c,0),0) for m in model_names_avail}} for c in channels]
        comp_pct=pd.DataFrame(rows_pct)
        comp_rev=pd.DataFrame(rows_rev)
        st.dataframe(comp_pct,use_container_width=True,hide_index=True)
        st.markdown(f'<p class="section-title">Revenue atribuido ({curr_sym})</p>',unsafe_allow_html=True)
        st.dataframe(comp_rev,use_container_width=True,hide_index=True)

        # ── Model reference cards ─────────────────────────────────────────────
        st.divider()
        st.markdown('<p class="section-title">Referencia de modelos</p>',unsafe_allow_html=True)
        cards_data=[
            ("First Click",   "#3b82f6","Regla",
             "100% del crédito al primer touchpoint del journey.",
             "Campañas de awareness; medir qué canal inicia el descubrimiento.",
             "Ignora completamente el canal que cierra la venta."),
            ("Last Click",    "#10b981","Regla",
             "100% del crédito al último touchpoint antes de la conversión.",
             "Campañas de performance directo; optimizar el canal de cierre.",
             "Subestima canales de nurturing y discovery."),
            ("Lineal",        "#f59e0b","Regla",
             "Crédito distribuido equitativamente entre todos los touchpoints.",
             "Sin hipótesis sobre qué canal importa más; etapa inicial (<500 conv.).",
             "No diferencia el impacto real de cada touchpoint."),
            ("Time Decay",    "#ec4899","Regla",
             "Mayor peso a los touchpoints más cercanos a la conversión.",
             "Ciclos de compra cortos; productos de bajo ticket.",
             "Penaliza canales de awareness en journeys largos."),
            ("U-Shape",       "#06b6d4","Regla",
             "40% al primero, 40% al último, 20% distribuido en los intermedios.",
             "Discovery y cierre igualmente importantes; mix pagado/orgánico.",
             "Los pesos fijos (40/40/20) no se adaptan al comportamiento real."),
            ("Markov",        "#8b5cf6","Data-Driven",
             "Mide cuánto cae la tasa de conv. al eliminar cada canal (removal effect).",
             "E-commerce con +2,000 journeys; mix de canales orgánico/pagado.",
             "Requiere volumen suficiente; puede ser lento con muchos canales."),
            ("Data-Driven",   "#ef4444","Data-Driven",
             "Teoría de juegos (Shapley): contribución marginal promedio de cada canal.",
             "Máxima precisión; canales correlacionados; +1,500 journeys.",
             "Lento con >8 canales; tiempo crece exponencialmente."),
        ]
        cards_html='<div style="display:flex;flex-wrap:wrap;gap:.6rem;">'
        for name,color,badge,logic,when,limit in cards_data:
            badge_bg="#dbeafe" if badge=="Regla" else "#ede9fe"
            badge_fg="#1e40af" if badge=="Regla" else "#6d28d9"
            cards_html+=f"""
            <div class="model-card" style="flex:1 1 210px;">
              <div style="margin-bottom:.45rem;">
                <span style="display:inline-block;width:10px;height:10px;background:{color};
                  border-radius:50%;margin-right:5px;vertical-align:middle;"></span>
                <b>{name}</b>&nbsp;
                <span style="background:{badge_bg};color:{badge_fg};border-radius:4px;
                  padding:2px 7px;font-size:.72rem;font-weight:600;">{badge}</span>
              </div>
              <div style="font-size:.81rem;color:#475569;margin-bottom:.2rem;">
                <b>Lógica:</b> {logic}</div>
              <div style="font-size:.81rem;color:#475569;margin-bottom:.2rem;">
                <b>Usar cuando:</b> {when}</div>
              <div style="font-size:.81rem;color:#dc2626;">
                <b>Limitación:</b> {limit}</div>
            </div>"""
        cards_html+='</div>'
        st.markdown(cards_html,unsafe_allow_html=True)

        # Export
        st.divider()
        exp2,_=st.columns([1,3])
        with exp2:
            pct_fmts={"Canal":"text",**{m:"pct" for m in model_names_avail}}
            rev_fmts={"Canal":"text",**{f"{m} ({curr_sym})":"money" for m in model_names_avail}}
            sheets2={
                "Crédito (%)":  (comp_pct,pct_fmts,"Crédito por canal — todos los modelos (%)"),
                "Crédito rev.": (comp_rev,rev_fmts,f"Revenue atribuido ({curr_sym})"),
            }
            buf2=build_excel(sheets2)
            st.download_button("⬇️ Descargar modelos de atribución (.xlsx)",buf2,"atribucion.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"Error en modelos de atribución: {e}")
        st.exception(e)

# ═══════════════════════════════════════════
# TAB 3
# ═══════════════════════════════════════════
with tab3:
    try:
        s_res=compute_simple_models(df,channels,direct_chs)
        s_pct={m:to_pct(s_res[m],channels) for m in s_res}
        all_res3=dict(s_res)
        all_pct3=dict(s_pct)
        if st.session_state.get("markov_result") is not None:
            mk=st.session_state["markov_result"]
            for c in channels: mk.setdefault(c,0)
            all_res3["Markov"]=mk
            all_pct3["Markov"]=to_pct(mk,channels)
        if st.session_state.get("shapley_result") is not None:
            sh=st.session_state["shapley_result"]
            for c in channels: sh.setdefault(c,0)
            all_res3["Data-Driven"]=sh
            all_pct3["Data-Driven"]=to_pct(sh,channels)
        model_names3=list(all_res3.keys())

        advanced_not_calc=[m for m in ["Markov","Data-Driven"] if m not in model_names3]
        if advanced_not_calc:
            st.markdown(f'<div class="warn-box">Modelos avanzados no calculados: <b>{" y ".join(advanced_not_calc)}</b>. Ve a "Modelos de atribución" para activarlos con sus botones.</div>',
                        unsafe_allow_html=True)

        paid_chs =[c for c in channels if c not in direct_chs and c not in organic_chs]
        seo_chs  =[c for c in channels if c in organic_chs]
        inv_chs  =paid_chs+seo_chs

        def ch_badge(c):
            if c in direct_chs:  return "🔴"
            if c in organic_chs: return "🟡"
            return "🔵"

        total_budget=budget+seo_budget
        legend_parts=[]
        if paid_chs:  legend_parts.append(f"🔵 Pagado: **{curr_sym} {budget:,.0f}**")
        if seo_chs:   legend_parts.append(f"🟡 SEO: **{curr_sym} {seo_budget:,.0f}**")
        if direct_chs:legend_parts.append(f"🔴 Directo: referencia sin inversión")
        st.markdown("  ·  ".join(legend_parts))
        st.markdown('<div class="info-box">Los sliders de canales <b>pagados</b> distribuyen el presupuesto publicitario. Los canales SEO usan su propio presupuesto de contenido.</div>',
                    unsafe_allow_html=True)

        sc1,sc2=st.columns([1,2])
        with sc1:
            scenario_name=st.text_input("Nombre del escenario","Escenario A")
            base_model=st.selectbox("Modelo base",model_names3,
                index=model_names3.index("Last Non-Direct") if "Last Non-Direct" in model_names3 else len(model_names3)-1)
            active_chs=st.multiselect("Canales activos",channels,
                default=[c for c in inv_chs if c in channels],
                help="Directo excluido por defecto — puedes añadirlo como referencia")

            if not active_chs:
                st.warning("Selecciona al menos un canal.")
                st.stop()

            base_pcts=all_pct3[base_model]
            base_active_tot=sum(base_pcts.get(c,0) for c in active_chs)
            base_norm={c:(base_pcts.get(c,0)/base_active_tot*100 if base_active_tot>0 else 0) for c in active_chs}

            active_paid=[c for c in active_chs if c not in organic_chs and c not in direct_chs]
            active_seo =[c for c in active_chs if c in organic_chs]

            weights={}
            if active_paid:
                st.markdown("**Canales pagados — % del presupuesto publicitario**")
                for c in active_paid:
                    paid_tot=sum(base_pcts.get(x,0) for x in active_paid)
                    default_w=base_pcts.get(c,0)/paid_tot*100 if paid_tot>0 else 0
                    weights[c]=st.slider(f"🔵 {c}",0.0,100.0,float(round(default_w,1)),step=0.5,format="%.1f%%")
            if active_seo:
                st.markdown("**Canales SEO — % del presupuesto SEO**")
                for c in active_seo:
                    seo_tot=sum(base_pcts.get(x,0) for x in active_seo)
                    default_w=base_pcts.get(c,0)/seo_tot*100 if seo_tot>0 else 0
                    weights[c]=st.slider(f"🟡 {c}",0.0,100.0,float(round(default_w,1)),step=0.5,format="%.1f%%")
            active_direct=[c for c in active_chs if c in direct_chs]

            paid_w_tot=sum(weights.get(c,0) for c in active_paid)
            seo_w_tot =sum(weights.get(c,0) for c in active_seo)
            alloc={}
            norm={}
            for c in active_paid:
                norm[c]=weights.get(c,0)/paid_w_tot*100 if paid_w_tot>0 else 0
                alloc[c]=norm[c]/100*budget
            for c in active_seo:
                norm[c]=weights.get(c,0)/seo_w_tot*100 if seo_w_tot>0 else 0
                alloc[c]=norm[c]/100*seo_budget
            for c in active_direct:
                norm[c]=0; alloc[c]=0

        with sc2:
            plot_chs=[c for c in active_chs if c not in active_direct]
            if plot_chs:
                fig_sc=go.Figure([
                    go.Bar(name=f"Modelo base ({base_model})",x=plot_chs,
                           y=[round(base_norm.get(c,0),1) for c in plot_chs],
                           marker_color="#94a3b8",opacity=0.7,
                           text=[f"{base_norm.get(c,0):.1f}%" for c in plot_chs],textposition="outside"),
                    go.Bar(name=scenario_name,x=plot_chs,
                           y=[round(norm.get(c,0),1) for c in plot_chs],
                           marker_color="#3b82f6",
                           text=[f"{norm.get(c,0):.1f}%" for c in plot_chs],textposition="outside"),
                ])
                fig_sc.update_layout(barmode="group",height=280,
                    yaxis=dict(title="% dentro de su tipo",ticksuffix="%"),
                    plot_bgcolor="white",paper_bgcolor="white",
                    legend=dict(orientation="h",y=1.12),margin=dict(l=0,r=0,t=40,b=8))
                st.plotly_chart(fig_sc,use_container_width=True)

            if active_paid and active_seo:
                d1,d2=st.columns(2)
                with d1:
                    st.caption(f"Pagado — {curr_sym} {budget:,.0f}")
                    fp=go.Figure(go.Pie(labels=active_paid,
                        values=[round(alloc[c],0) for c in active_paid],
                        hole=0.55,marker_colors=PALETTE[:len(active_paid)],
                        textinfo="label+percent",
                        hovertemplate="%{label}<br>%{value:,.0f} "+curr_sym+"<extra></extra>"))
                    fp.update_layout(height=230,margin=dict(l=0,r=0,t=5,b=5),paper_bgcolor="white",
                        annotations=[dict(text=f"{curr_sym}\n{budget:,.0f}",x=0.5,y=0.5,font_size=11,showarrow=False)])
                    st.plotly_chart(fp,use_container_width=True)
                with d2:
                    st.caption(f"SEO — {curr_sym} {seo_budget:,.0f}")
                    fs=go.Figure(go.Pie(labels=active_seo,
                        values=[round(alloc[c],0) for c in active_seo],
                        hole=0.55,marker_colors=["#10b981","#34d399","#6ee7b7"][:len(active_seo)],
                        textinfo="label+percent",
                        hovertemplate="%{label}<br>%{value:,.0f} "+curr_sym+"<extra></extra>"))
                    fs.update_layout(height=230,margin=dict(l=0,r=0,t=5,b=5),paper_bgcolor="white",
                        annotations=[dict(text=f"{curr_sym}\n{seo_budget:,.0f}",x=0.5,y=0.5,font_size=11,showarrow=False)])
                    st.plotly_chart(fs,use_container_width=True)
            else:
                donut_chs=[c for c in active_chs if c not in active_direct]
                total_shown=budget if not active_seo else seo_budget
                fig_donut=go.Figure(go.Pie(
                    labels=donut_chs,values=[round(alloc.get(c,0),0) for c in donut_chs],
                    hole=0.55,marker_colors=[PALETTE[i%len(PALETTE)] for i in range(len(donut_chs))],
                    textinfo="label+percent",
                    hovertemplate="%{label}<br>%{value:,.0f} "+curr_sym+"<extra></extra>"))
                fig_donut.update_layout(height=260,margin=dict(l=0,r=0,t=8,b=8),paper_bgcolor="white",
                    annotations=[dict(text=f"{curr_sym}<br>{total_shown:,.0f}",x=0.5,y=0.5,font_size=13,showarrow=False)])
                st.plotly_chart(fig_donut,use_container_width=True)

            bud_rows=[]
            for c in active_chs:
                tipo="Directo" if c in direct_chs else ("SEO" if c in organic_chs else "Pagado")
                delta=round(norm.get(c,0)-base_norm.get(c,0),1)
                bud_rows.append({"Canal":c,"Tipo":tipo,
                    f"Base {base_model} (%)":round(base_norm.get(c,0),1),
                    "Escenario (%)":round(norm.get(c,0),1),
                    f"Inversión ({curr_sym})":round(alloc.get(c,0),0),
                    "Δ (pp)":f"{'+'if delta>=0 else ''}{delta}"})
            bud_df=pd.DataFrame(bud_rows)
            st.dataframe(bud_df,use_container_width=True,hide_index=True)

        st.divider()
        st.markdown('<p class="section-title">Comparar inversión por modelo — solo canales invertibles</p>',unsafe_allow_html=True)
        inv_channels=[c for c in channels if c not in direct_chs]
        multi_rows=[]
        for c in inv_channels:
            tipo="SEO" if c in organic_chs else "Pagado"
            ch_budget=seo_budget if c in organic_chs else budget
            row={"Canal":c,"Tipo":tipo}
            for m in model_names3:
                p_act={ch:all_pct3[m].get(ch,0) for ch in inv_channels}
                tot=sum(p_act.values())
                row[f"{m} ({curr_sym})"]=round(p_act[c]/tot*ch_budget if tot>0 else 0,0)
            multi_rows.append(row)
        multi_df=pd.DataFrame(multi_rows)

        fig_multi=go.Figure()
        for m in model_names3:
            fig_multi.add_trace(go.Bar(name=m,x=multi_df["Canal"],y=multi_df[f"{m} ({curr_sym})"],
                marker_color=MODEL_COLORS.get(m,PALETTE[0]),
                text=[f"{curr_sym} {int(vv):,}" for vv in multi_df[f"{m} ({curr_sym})"]],
                textposition="outside"))
        fig_multi.update_layout(barmode="group",height=320,
            yaxis_title=f"Inversión ({curr_sym})",
            plot_bgcolor="white",paper_bgcolor="white",
            legend=dict(orientation="h",y=1.1),margin=dict(l=0,r=0,t=40,b=8))
        st.plotly_chart(fig_multi,use_container_width=True)
        st.dataframe(multi_df,use_container_width=True,hide_index=True)

        st.divider()
        exp3a,exp3b,_=st.columns([1,1,2])
        with exp3a:
            bud_exp=bud_df.copy()
            for col in [f"Base {base_model} (%)","Escenario (%)",f"Inversión ({curr_sym})"]:
                bud_exp[col]=pd.to_numeric(bud_exp[col],errors="coerce")
            bud_fmts={"Canal":"text",f"Base {base_model} (%)":"pct","Escenario (%)":"pct",
                      f"Inversión ({curr_sym})":"money","Δ (pp)":"text"}
            multi_fmts={"Canal":"text",**{f"{m} ({curr_sym})":"money" for m in model_names3}}
            sheets3={"Escenario activo":(bud_exp,bud_fmts,f"Escenario: {scenario_name}"),
                     "Multi-modelo":    (multi_df,multi_fmts,f"Inversión por modelo — {curr_sym} {budget:,.0f}")}
            buf3=build_excel(sheets3)
            st.download_button("⬇️ Descargar escenarios (.xlsx)",buf3,"escenarios.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with exp3b:
            rows_pct_all=[{"Canal":c,**{m:round(all_pct3[m][c],1) for m in model_names3}} for c in channels]
            rows_rev_all=[{"Canal":c,**{f"{m} ({curr_sym})":round(all_res3[m].get(c,0),0) for m in model_names3}} for c in channels]
            all_sheets={
                "KPIs":(pd.DataFrame({"Métrica":["Journeys","Conversiones","Tasa conv. (%)","Revenue total","Ticket prom."],
                                       "Valor":[n_total,n_conv,round(conv_rate,1),round(total_rev,0),round(avg_order,0)]}),
                        {"Métrica":"text","Valor":"text"},"Resumen del dataset"),
                "Frecuencia canales":(pd.DataFrame({"Canal":list(ch_cnt.keys()),"Touchpoints":list(ch_cnt.values())}).sort_values("Touchpoints",ascending=False),
                                      {"Canal":"text","Touchpoints":"int"},"Touchpoints por canal"),
                "Crédito (%)":(pd.DataFrame(rows_pct_all),{"Canal":"text",**{m:"pct" for m in model_names3}},"Crédito por canal (%)"),
                "Crédito rev.":(pd.DataFrame(rows_rev_all),{"Canal":"text",**{f"{m} ({curr_sym})":"money" for m in model_names3}},f"Revenue atribuido ({curr_sym})"),
                "Escenario activo":(bud_exp,bud_fmts,f"Escenario: {scenario_name}"),
                "Multi-modelo":(multi_df,multi_fmts,"Inversión por modelo"),
            }
            buf_all=build_excel(all_sheets)
            st.download_button("⬇️ Reporte completo (.xlsx)",buf_all,"attribution_reporte_completo.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",type="primary")
    except Exception as e:
        st.error(f"Error en escenarios: {e}")
        st.exception(e)

# ═══════════════════════════════════════════
# TAB 4 — Guía: preparar datos
# ═══════════════════════════════════════════
with tab4:
    try:
        st.markdown("## 📋 Guía: preparar datos")

        # ── Sección 1 ─────────────────────────────────────────────────────────
        st.markdown('<p class="section-title">1. Estructura mínima del CSV</p>',unsafe_allow_html=True)
        ejemplo_csv=pd.DataFrame({
            "journey_id":[1,2,3,4,5],
            "path":[
                "Organic Search",
                "Paid Search > Email",
                "Social Media > Paid Search > Email",
                "Direct > Organic Search",
                "Paid Search > Social Media > Email > Direct",
            ],
            "converted":[1,1,1,0,1],
            "revenue":[250.0,120.0,450.0,0.0,310.0],
        })
        st.dataframe(ejemplo_csv,use_container_width=True,hide_index=True)
        st.markdown("""
<div class="info-box">
<b>Reglas del formato:</b><br>
• El separador entre canales debe ser <code> &gt; </code> (con espacios a cada lado)<br>
• Los nombres de canales deben ser consistentes (mismo case, sin espacios extra)<br>
• <code>revenue</code> debe ser 0 si <code>converted</code> = 0
</div>""",unsafe_allow_html=True)

        # ── Sección 2 ─────────────────────────────────────────────────────────
        st.markdown('<p class="section-title">2. ¿Qué es un journey y ventana de atribución?</p>',unsafe_allow_html=True)
        st.markdown("""
Un **journey** es la secuencia de canales que un usuario tocó antes de convertir (o no) dentro
de un período de tiempo definido. La **ventana de atribución** determina cuánto tiempo atrás
contar los touchpoints:
""")
        win_df=pd.DataFrame({
            "Tipo de negocio":["E-commerce bajo ticket (≤ S/300)","E-commerce medio-alto ticket","B2B o servicios financieros"],
            "Ventana recomendada":["7 – 14 días","30 – 60 días","Hasta 90 días"],
        })
        st.dataframe(win_df,use_container_width=True,hide_index=True)

        # ── Sección 3 ─────────────────────────────────────────────────────────
        st.markdown('<p class="section-title">3. Fuentes de datos</p>',unsafe_allow_html=True)
        with st.expander("GA4 + BigQuery (recomendado)"):
            st.markdown("Exporta los eventos de GA4 a BigQuery y construye los journeys con esta query base:")
            st.code("""SELECT user_pseudo_id, event_timestamp,
  traffic_source.medium AS canal
FROM `proyecto.analytics_XXXX.events_*`
WHERE event_name IN ('session_start','purchase')
ORDER BY user_pseudo_id, event_timestamp""", language="sql")
        with st.expander("Google Ads + GA4 auto-tagging"):
            st.markdown("""
Activa el **auto-tagging** en Google Ads para que los parámetros `gclid` se capturen en GA4.
Usa el campo `traffic_source.medium` = `'cpc'` para identificar tráfico pagado de Google.
Combínalo con dimensiones de campaña exportando desde la API de Google Ads si necesitas
granularidad por campaña o grupo de anuncios.
""")
        with st.expander("CRM / Email (Mailchimp, HubSpot)"):
            st.markdown("""
Usa UTMs en todos los enlaces de email: `utm_source=mailchimp&utm_medium=email&utm_campaign=nombre`.
En GA4, el canal aparecerá como `email` en `traffic_source.medium`.
Para journeys cross-dispositivo, cruza por `user_id` si tienes login implementado.
""")
        with st.expander("Meta Ads"):
            st.markdown("""
Implementa la **API de Conversiones de Meta** (server-side) para reducir pérdida de datos por
bloqueadores de ads. Usa UTMs estándar: `utm_source=facebook&utm_medium=paid_social`.
Los datos de Meta no reflejan el journey completo — combínalos siempre con GA4.
""")

        # ── Sección 4 ─────────────────────────────────────────────────────────
        st.markdown('<p class="section-title">4. Script Python — construcción desde BigQuery</p>',unsafe_allow_html=True)
        st.code("""from google.cloud import bigquery
import pandas as pd
from datetime import date, timedelta

PROJECT   = "mi-proyecto-gcp"
DATASET   = "analytics_12345678"   # Dataset ID de GA4
WINDOW    = 30                     # ventana de atribución en días

client      = bigquery.Client(project=PROJECT)
fecha_fin   = date.today().strftime("%Y%m%d")
fecha_ini   = (date.today() - timedelta(days=90)).strftime("%Y%m%d")

QUERY = f\"\"\"
WITH eventos AS (
  SELECT
    user_pseudo_id,
    TIMESTAMP_MICROS(event_timestamp)          AS ts,
    COALESCE(traffic_source.medium, 'direct')  AS canal,
    event_name,
    COALESCE((SELECT value.int_value
              FROM UNNEST(event_params)
              WHERE key = 'value'), 0)          AS revenue
  FROM `{PROJECT}.{DATASET}.events_*`
  WHERE event_name IN ('session_start', 'purchase')
    AND _TABLE_SUFFIX BETWEEN '{fecha_ini}' AND '{fecha_fin}'
),
journeys AS (
  SELECT
    user_pseudo_id,
    STRING_AGG(canal, ' > ' ORDER BY ts)           AS path,
    MAX(CASE WHEN event_name='purchase' THEN 1
             ELSE 0 END)                           AS converted,
    SUM(CASE WHEN event_name='purchase' THEN revenue
             ELSE 0 END)                           AS revenue
  FROM eventos
  GROUP BY user_pseudo_id
  HAVING TIMESTAMP_DIFF(MAX(ts), MIN(ts), DAY) <= {WINDOW}
)
SELECT
  ROW_NUMBER() OVER (ORDER BY user_pseudo_id) AS journey_id,
  path,
  converted,
  ROUND(CAST(revenue AS FLOAT64), 2)           AS revenue
FROM journeys
WHERE path IS NOT NULL
\"\"\"

df = client.query(QUERY).to_dataframe()
df = df[df["path"].notna() & (df["path"] != "")]
df.to_csv("journeys_atribucion.csv", index=False)
print(f"✓ {len(df):,} journeys | {int(df['converted'].sum()):,} conversiones")
""", language="python")

        # ── Sección 5 ─────────────────────────────────────────────────────────
        st.markdown('<p class="section-title">5. Volumen mínimo por modelo</p>',unsafe_allow_html=True)
        vol_df=pd.DataFrame({
            "Modelo":["First Click / Last Click","Lineal / Time Decay / U-Shape","Markov","Data-Driven (Shapley)"],
            "Journeys mínimos":["500+","1,000+","2,000+","1,500+"],
            "Conversiones mínimas":["200+","400+","800+","500+"],
            "Canales máx. recomendados":["Sin límite","Sin límite","10","8"],
        })
        st.dataframe(vol_df,use_container_width=True,hide_index=True)

        # ── Sección 6 ─────────────────────────────────────────────────────────
        st.markdown('<p class="section-title">6. Errores comunes</p>',unsafe_allow_html=True)
        with st.expander("⚠️ Separador incorrecto (ej. '|' o ',' en lugar de ' > ')"):
            st.markdown("Los canales no se detectan correctamente. Solución:")
            st.code("""df["path"] = df["path"].str.replace("|", " > ", regex=False)
df["path"] = df["path"].str.replace(",", " > ", regex=False)""", language="python")

        with st.expander("⚠️ Nombres de canales inconsistentes ('paid search' vs 'Paid Search')"):
            st.markdown("Genera canales duplicados con diferente capitalización. Solución:")
            st.code("""df["path"] = df["path"].apply(
    lambda p: " > ".join(c.strip().title() for c in p.split(">"))
)""", language="python")

        with st.expander("⚠️ Revenue distinto de cero en journeys no convertidos"):
            st.markdown("Distorsiona todos los modelos basados en revenue. Verificación:")
            st.code("""# Detectar inconsistencias
mask = (df["converted"] == 0) & (df["revenue"] > 0)
print(f"Filas con problema: {mask.sum()}")
# Corrección
df.loc[df["converted"] == 0, "revenue"] = 0""", language="python")

        with st.expander("⚠️ Journeys demasiado largos (ventana de atribución muy amplia)"):
            st.markdown("Incluye touchpoints irrelevantes de meses anteriores. Filtro por fecha:")
            st.code("""from datetime import timedelta
# Asumiendo que tienes columnas first_touch y last_touch como datetime
df["window_days"] = (df["last_touch"] - df["first_touch"]).dt.days
df = df[df["window_days"] <= 30]  # ventana de 30 días""", language="python")

        with st.expander("⚠️ Canales duplicados consecutivos ('Email > Email > Email')"):
            st.markdown("Infla artificialmente la frecuencia del canal. Deduplicación:")
            st.code("""import re
def dedup_path(path):
    parts = [c.strip() for c in path.split(">")]
    deduped = [parts[0]] + [b for a, b in zip(parts, parts[1:]) if a != b]
    return " > ".join(deduped)

df["path"] = df["path"].apply(dedup_path)""", language="python")

    except Exception as e:
        st.error(f"Error en guía de preparar datos: {e}")
        st.exception(e)

# ═══════════════════════════════════════════
# TAB 5 — Guía: interpretar resultados
# ═══════════════════════════════════════════
with tab5:
    try:
        st.markdown("## 📖 Guía: interpretar resultados")

        # ── Sección 1 ─────────────────────────────────────────────────────────
        st.markdown('<p class="section-title">1. Cómo leer el análisis descriptivo</p>',unsafe_allow_html=True)
        guide1_items=[
            ("Tasa de conversión general",
             "Rango normal en e-commerce: 1–5%. Si supera el 60%, probablemente hay sesiones post-conversión incluidas en los journeys — revisa que el evento de conversión no se dispare en páginas de confirmación.",),
            ("Frecuencia de touchpoints por canal",
             "Alta frecuencia ≠ alta importancia. Un canal puede aparecer mucho (como Direct) sin generar conversiones. Compara siempre con los modelos de atribución para entender el rol real de cada canal.",),
            ("Longitud del journey",
             "Si más del 60% de los journeys tienen 1 solo touchpoint, los modelos avanzados (Markov, Shapley) aportarán poco — todos los créditos caerán en un solo canal. Verifica que la ventana de atribución sea suficientemente larga.",),
            ("Tasa de conv. primer vs. último canal",
             "Canal con tasa alta como 'primer canal' = canal de discovery (genera demanda). Canal con tasa alta como 'último canal' = canal de cierre. Esta distinción guía cuándo usar First Click vs Last Click.",),
            ("Distribución del revenue",
             "Una distribución muy sesgada a la derecha (pocos pedidos de ticket muy alto) indica outliers. Considera si esos pedidos son representativos o si distorsionan los modelos que usan revenue para asignar crédito.",),
        ]
        for title,body in guide1_items:
            st.markdown(f'<div class="guide-box"><b>{title}</b><br><span style="font-size:.88rem;color:#475569;">{body}</span></div>',
                        unsafe_allow_html=True)

        # ── Sección 2 ─────────────────────────────────────────────────────────
        st.markdown('<p class="section-title">2. Cómo leer el heatmap de modelos</p>',unsafe_allow_html=True)
        st.markdown("""<div class="guide-box">
<b>Azul oscuro</b> = mucho crédito asignado &nbsp;|&nbsp; <b>Azul claro</b> = poco crédito<br><br>
<b>Patrones clave:</b><br>
• <b>First Click y Last Click muy distintos</b> → los canales de discovery y de cierre son diferentes; considera U-Shape o Time Decay.<br>
• <b>Markov y Data-Driven coinciden</b> → consenso estadístico robusto; esos valores son confiables para tomar decisiones de inversión.<br>
• <b>Crédito alto solo en Last Click</b> → canal de cierre que depende del trabajo previo de otros canales; no lo escales solo.<br>
• <b>Crédito consistente en todos los modelos</b> → canal crítico y robusto; reducirlo tiene alto riesgo.
</div>""",unsafe_allow_html=True)

        # ── Sección 3 ─────────────────────────────────────────────────────────
        st.markdown('<p class="section-title">3. Qué modelo elegir según escenario</p>',unsafe_allow_html=True)
        decision_rows=[
            ("Solo optimizas conversiones directas","Last Click","Simple, auditable, alineado a plataformas de ads."),
            ("Mides impacto de brand / awareness","First Click o U-Shape","Valora el canal que inicia el descubrimiento."),
            ("Journeys 3+ pasos, mix pagado/orgánico","Time Decay o U-Shape","Reconoce el funnel completo sin ignorar el inicio."),
            ("E-commerce con +2,000 journeys","Markov","Data-driven sin supuestos fijos; usa los datos reales."),
            ("Máxima precisión estadística","Data-Driven (Shapley)","Más robusto ante canales correlacionados."),
            ("Etapa inicial, <500 conversiones","Lineal","Evita sesgos por falta de datos; base neutral."),
        ]
        table_html='<div style="border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;font-size:.85rem;">'
        table_html+='<div style="display:flex;background:#1e40af;color:#fff;font-weight:600;padding:.55rem 1rem;gap:1rem;">'
        table_html+='<div style="flex:2.2;">Escenario</div><div style="flex:1.5;">Modelo recomendado</div><div style="flex:2.5;">Razón</div></div>'
        for i,(esc,mod,raz) in enumerate(decision_rows):
            bg="#fff" if i%2==0 else "#f8fafc"
            table_html+=f'<div style="display:flex;padding:.55rem 1rem;gap:1rem;background:{bg};border-top:1px solid #e2e8f0;">'
            table_html+=f'<div style="flex:2.2;">{esc}</div><div style="flex:1.5;font-weight:600;">{mod}</div><div style="flex:2.5;color:#475569;">{raz}</div></div>'
        table_html+='</div>'
        st.markdown(table_html,unsafe_allow_html=True)

        # ── Sección 4 ─────────────────────────────────────────────────────────
        st.markdown('<p class="section-title">4. Del % de crédito a la inversión real</p>',unsafe_allow_html=True)
        st.markdown("""<div class="guide-box">
<b>Cuatro principios para traducir atribución en presupuesto:</b><br><br>
<b>1. Rendimientos decrecientes</b> — No concentres más del 50% del presupuesto en un solo canal,
aunque tenga el mayor crédito. El rendimiento marginal cae al escalar.<br><br>
<b>2. Cobertura de funnel</b> — Mantén siempre una inversión base en canales de discovery
(alta tasa como primer canal). Pausarlos destruye el pipeline futuro aunque no se vea
en conversiones inmediatas.<br><br>
<b>3. Eficiencia</b> — Combina el % de crédito con el CPA o CPL real de cada canal.
Un canal con 30% de crédito y CPA bajo es mejor candidato para escalar que uno con 40%
de crédito y CPA muy alto.<br><br>
<b>4. Regla práctica</b> — Usa los porcentajes de atribución como base de distribución,
pero aplica un tope máximo por canal (ej. 40%) para mantener diversificación y resiliencia.
</div>""",unsafe_allow_html=True)

        # ── Sección 5 ─────────────────────────────────────────────────────────
        st.markdown('<p class="section-title">5. Señales de alerta</p>',unsafe_allow_html=True)
        with st.expander("⚠️ Markov y Data-Driven muestran resultados muy distintos"):
            st.markdown("""
Puede indicar volumen insuficiente para que los valores converjan, o canales altamente
correlacionados (ej. Paid Search y Retargeting siempre aparecen juntos).
**Acción:** Aumenta la ventana de datos o agrupa canales relacionados antes de recalcular.
""")
        with st.expander("⚠️ Un canal muestra 0% en todos los modelos"):
            st.markdown("""
El canal probablemente tiene un nombre diferente en el CSV al esperado.
**Acción:** Revisa la columna `path` del CSV con `df['path'].str.split(' > ').explode().value_counts()`
y verifica que el nombre del canal sea exactamente el mismo en todos los journeys.
""")
        with st.expander("⚠️ Tasa de conversión superior al 60%"):
            st.markdown("""
Señal clara de que las sesiones post-conversión (ej. página de confirmación, thank-you page)
están siendo contadas como journeys adicionales.
**Acción:** Filtra para que cada `journey_id` solo contenga eventos anteriores al primer evento
de conversión.
""")
        with st.expander("⚠️ First Click y Last Click producen resultados idénticos"):
            st.markdown("""
La mayoría de journeys tienen un solo touchpoint, por lo que el primer y último canal son el mismo.
Los modelos avanzados aportarán poco valor en este caso.
**Acción:** Amplía la ventana de atribución y verifica que `session_start` se esté registrando
correctamente en GA4.
""")
        with st.expander("⚠️ Markov tarda más de 60 segundos"):
            st.markdown("""
Con muchos canales o un dataset muy grande, las iteraciones de la cadena de Markov se vuelven lentas.
**Acciones posibles:**
- Limita el path a los últimos 5 touchpoints antes de la conversión.
- Agrupa canales similares (ej. "Paid Search Brand" + "Paid Search Generic" → "Paid Search").
- Filtra journeys con muy baja frecuencia (menos de 10 apariciones).
""")

        # ── Sección 6 ─────────────────────────────────────────────────────────
        st.markdown('<p class="section-title">6. Framework de entregable al cliente</p>',unsafe_allow_html=True)
        st.markdown("""<div class="guide-box">
<b>5 pasos para estructurar el análisis:</b><br><br>
<b>1. Diagnóstico (descriptivo)</b> — Presenta las métricas clave: tasa de conversión,
longitud media del journey, canales más frecuentes. Establece el contexto del negocio.<br><br>
<b>2. Análisis de atribución (2-3 modelos)</b> — Elige los modelos más relevantes
para el cliente (ej. Last Click + U-Shape + Markov). Muestra el heatmap y explica
las diferencias entre modelos, no solo los números.<br><br>
<b>3. Recomendación de presupuesto con justificación</b> — Presenta la distribución
propuesta basada en el modelo elegido. Explica por qué cada canal recibe ese porcentaje
y cuál es la fuente de datos que lo respalda.<br><br>
<b>4. Plan de acción</b> — Define qué canal escalar, cuál optimizar y cuál pausar.
Incluye KPIs a 30 / 60 / 90 días (ej. CPA objetivo, ROAS mínimo, volumen de conversiones).<br><br>
<b>5. Próximo análisis</b> — Programa la revisión en 60-90 días con los mismos datos
actualizados para validar si la reasignación de presupuesto mejoró los resultados.
</div>""",unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error en guía de interpretar resultados: {e}")
        st.exception(e)
