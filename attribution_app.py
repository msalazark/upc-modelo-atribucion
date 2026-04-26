import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
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
</style>
""", unsafe_allow_html=True)

PALETTE = ["#3b82f6","#10b981","#f59e0b","#ec4899","#8b5cf6","#06b6d4","#ef4444","#84cc16"]
MODEL_COLORS = {"First Click":"#3b82f6","Last Click":"#10b981","Last Non-Direct":"#06b6d4",
                "Lineal":"#f59e0b","Time Decay":"#ec4899","Markov":"#8b5cf6"}
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
    """Last click ignorando canales directos/no invertibles."""
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

def time_decay(conv,decay=0.7):
    d={}
    for _,r in conv.iterrows():
        p=parse_path(r["path"]); v=float(r["revenue"]); n=len(p)
        if not n: continue
        w=[np.exp(decay*(i-n+1)) for i in range(n)]; ws=sum(w)
        for i,c in enumerate(p): d[c]=d.get(c,0)+v*w[i]/ws
    return d

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
            if rs>0: t2[s]={e:v/rs for e,v in t2[s].items()}
        v={s:(1.0 if s=="__conv__" else 0.0) for s in act}
        for _ in range(300):
            nv={s:v[s] for s in act}
            for s in act:
                if s in("__conv__","__null__"): continue
                nv[s]=sum(t2[s].get(e,0)*v.get(e,0) for e in act)
            v=nv
        return v.get("__start__",0)
    base=rate()
    eff={c:max(0,base-rate(c)) for c in channels}
    te=sum(eff.values())
    total_rev=float(df[df["converted"]==1]["revenue"].sum())
    return {c:(eff[c]/te*total_rev if te>0 else 0) for c in channels}

def to_pct(d,channels):
    tot=sum(d.get(c,0) for c in channels)
    return {c:(d.get(c,0)/tot*100 if tot>0 else 0) for c in channels}

def compute_simple_models(df, channels, direct_chs=None):
    conv=df[df["converted"]==1].copy()
    skip=set(direct_chs or [])
    res={"First Click":first_click(conv),"Last Click":last_click(conv),
         "Last Non-Direct":last_non_direct(conv,skip),
         "Lineal":linear_model(conv),"Time Decay":time_decay(conv)}
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

        # Channel classification (computed early from raw data)
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

    # Detect channels
    all_raw=[]
    for p in df["path"]: all_raw.extend(parse_path(p))
    channels=sorted(set(all_raw))
    if not channels:
        st.error("No se detectaron canales. Verifica que la columna 'path' use ' > ' como separador.")
        st.stop()

    df["n_tp"]    =df["path"].apply(lambda p:len(parse_path(p)))
    df["first_ch"]=df["path"].apply(lambda p:parse_path(p)[0] if parse_path(p) else None)
    df["last_ch"] =df["path"].apply(lambda p:parse_path(p)[-1] if parse_path(p) else None)

    conv_df   =df[df["converted"]==1].copy()
    total_rev =float(conv_df["revenue"].sum())
    n_conv    =int(df["converted"].sum())
    n_total   =len(df)
    conv_rate =n_conv/n_total*100 if n_total>0 else 0
    avg_order =total_rev/n_conv if n_conv>0 else 0

    # Channel frequency (global)
    ch_cnt=Counter(all_raw)

except Exception as e:
    st.error(f"Error procesando el dataset: {e}")
    st.exception(e)
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1,tab2,tab3=st.tabs(["📈  Análisis descriptivo","🏅  Modelos de atribución","🎯  Escenarios de inversión"])

# ═══════════════════════════════════════════
# TAB 1
# ═══════════════════════════════════════════
with tab1:
    try:
        # KPIs
        k1,k2,k3,k4,k5=st.columns(5)
        k1.metric("Journeys",f"{n_total:,}")
        k2.metric("Conversiones",f"{n_conv:,}")
        k3.metric("Tasa de conv.",f"{conv_rate:.1f}%")
        k4.metric(f"Revenue ({curr_sym})",f"{total_rev:,.0f}")
        k5.metric(f"Ticket prom. ({curr_sym})",f"{avg_order:,.0f}")
        st.divider()

        # Row 1
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

        # Row 2
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

        # Row 3: ticket promedio por canal
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

        # Top paths
        st.markdown('<p class="section-title">Top 10 journeys más frecuentes</p>',unsafe_allow_html=True)
        path_stats=df.groupby("path").agg(Journeys=("converted","count"),
            Conversiones=("converted","sum"),Revenue=("revenue","sum")).reset_index()
        path_stats["Conv. Rate (%)"]=( path_stats["Conversiones"]/path_stats["Journeys"]*100).round(1)
        path_stats["Revenue"]=path_stats["Revenue"].round(0)
        top_paths=path_stats.sort_values("Journeys",ascending=False).head(10).rename(columns={"path":"Journey path"})
        st.dataframe(top_paths,use_container_width=True,hide_index=True)

        # Export Tab1
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
        st.markdown('<div class="info-box">Los 4 primeros modelos se calculan al instante. <b>Markov</b> (data-driven) puede tardar 15-30 segundos con datasets grandes — actívalo con el botón.</div>',
                    unsafe_allow_html=True)

        # Compute simple models
        simple_res=compute_simple_models(df,channels,direct_chs)
        simple_pct={m:to_pct(simple_res[m],channels) for m in simple_res}

        # Markov
        run_markov=st.button("🔮 Calcular modelo Markov (data-driven)",type="primary")
        if "markov_result" not in st.session_state:
            st.session_state["markov_result"]=None

        if run_markov:
            with st.spinner("Calculando cadenas de Markov... (15-30 segundos)"):
                try:
                    mk=markov_model(df,channels)
                    st.session_state["markov_result"]=mk
                    st.success("✓ Markov calculado correctamente")
                except Exception as e:
                    st.error(f"Error en Markov: {e}")

        # Merge results
        all_res=dict(simple_res)
        all_pct=dict(simple_pct)
        if st.session_state["markov_result"] is not None:
            mk=st.session_state["markov_result"]
            for c in channels: mk.setdefault(c,0)
            all_res["Markov"]=mk
            all_pct["Markov"]=to_pct(mk,channels)
        model_names_avail=list(all_res.keys())

        # Grouped bar
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

        # Heatmap
        st.markdown('<p class="section-title">Mapa de calor — % de crédito</p>',unsafe_allow_html=True)
        z=[[round(all_pct[m][c],1) for c in channels] for m in model_names_avail]
        fig_heat=go.Figure(go.Heatmap(z=z,x=channels,y=model_names_avail,
            colorscale="Blues",text=[[f"{v:.1f}%" for v in row] for row in z],
            texttemplate="%{text}",showscale=True,colorbar=dict(title="%")))
        fig_heat.update_layout(height=max(220,len(model_names_avail)*50),
            margin=dict(l=0,r=0,t=8,b=8),plot_bgcolor="white",paper_bgcolor="white")
        st.plotly_chart(fig_heat,use_container_width=True)

        # Radar
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

        # Tables
        st.markdown('<p class="section-title">Tabla comparativa — crédito (%)</p>',unsafe_allow_html=True)
        rows_pct=[{"Canal":c,**{m:round(all_pct[m][c],1) for m in model_names_avail}} for c in channels]
        rows_rev=[{"Canal":c,**{f"{m} ({curr_sym})":round(all_res[m].get(c,0),0) for m in model_names_avail}} for c in channels]
        comp_pct=pd.DataFrame(rows_pct)
        comp_rev=pd.DataFrame(rows_rev)
        st.dataframe(comp_pct,use_container_width=True,hide_index=True)
        st.markdown('<p class="section-title">Revenue atribuido ({curr_sym})</p>',unsafe_allow_html=True)
        st.dataframe(comp_rev,use_container_width=True,hide_index=True)

        # Model guide
        st.divider()
        st.markdown('<p class="section-title">Guía rápida: ¿cuándo usar cada modelo?</p>',unsafe_allow_html=True)
        model_info=[
            ("First Click","#3b82f6","Premia el canal que generó la primera impresión. Ideal para awareness.","Ignora completamente el cierre de la venta."),
            ("Last Click","#10b981","Premia el canal final. Simple y auditable para campañas de cierre.","Subestima canales de nurturing y discovery."),
            ("Last Non-Direct","#06b6d4","Excluye canales directos/orgánicos: útil cuando el tráfico directo distorsiona el crédito.","Requiere clasificar correctamente los canales no invertibles."),
            ("Lineal","#f59e0b","Distribución equitativa. Base neutral sin supuestos fuertes.","No diferencia el impacto real de cada canal."),
            ("Time Decay","#ec4899","Mayor peso a touchpoints cercanos a la conversión. Ideal para ciclos cortos.","Penaliza canales de awareness en journeys largos."),
            ("Markov","#8b5cf6","Data-driven: mide cuánto cae la tasa de conv. al eliminar cada canal.","Requiere volumen suficiente; tarda más en calcular."),
        ]
        cols_m=st.columns(6)
        for col,(m,color,pro,con) in zip(cols_m,model_info):
            with col:
                st.markdown(f"<span style='display:inline-block;width:10px;height:10px;background:{color};border-radius:50%;margin-right:5px;vertical-align:middle;'></span>**{m}**",unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:.8rem;color:#15803d;'>✔ {pro}</span>",unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:.8rem;color:#b91c1c;'>✖ {con}</span>",unsafe_allow_html=True)

        # Export Tab 2
        st.divider()
        exp2,_=st.columns([1,3])
        with exp2:
            pct_fmts={"Canal":"text",**{m:"pct" for m in model_names_avail}}
            rev_fmts={"Canal":"text",**{f"{m} ({curr_sym})":"money" for m in model_names_avail}}
            sheets2={
                "Crédito (%)":   (comp_pct,pct_fmts,"Crédito por canal — todos los modelos (%)"),
                "Crédito rev.":  (comp_rev,rev_fmts,f"Revenue atribuido ({curr_sym})"),
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
        # Reload simple models (always available)
        s_res=compute_simple_models(df,channels,direct_chs)
        s_pct={m:to_pct(s_res[m],channels) for m in s_res}
        all_res3=dict(s_res)
        all_pct3=dict(s_pct)
        if st.session_state.get("markov_result") is not None:
            mk=st.session_state["markov_result"]
            for c in channels: mk.setdefault(c,0)
            all_res3["Markov"]=mk
            all_pct3["Markov"]=to_pct(mk,channels)
        model_names3=list(all_res3.keys())

        if "Markov" not in model_names3:
            st.markdown('<div class="warn-box">El modelo Markov no está calculado. Ve a "Modelos de atribución" y presiona "Calcular modelo Markov".</div>',
                        unsafe_allow_html=True)

        # Channel type classification for this tab
        paid_chs =[c for c in channels if c not in direct_chs and c not in organic_chs]
        seo_chs  =[c for c in channels if c in organic_chs]
        inv_chs  =paid_chs+seo_chs  # investable channels (default active)

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

            # Paid channels → use advertising budget
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
            # Direct channels shown as reference (no slider, no budget)
            active_direct=[c for c in active_chs if c in direct_chs]

            # Normalize and allocate per type
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
            # Bar: only paid+seo channels (exclude reference direct)
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

            # Two donuts side by side if both types present
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

            # Table
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

        # Multi-model comparison (only investable channels)
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
                text=[f"{curr_sym} {int(v):,}" for v in multi_df[f"{m} ({curr_sym})"]],
                textposition="outside"))
        fig_multi.update_layout(barmode="group",height=320,
            yaxis_title=f"Inversión ({curr_sym})",
            plot_bgcolor="white",paper_bgcolor="white",
            legend=dict(orientation="h",y=1.1),margin=dict(l=0,r=0,t=40,b=8))
        st.plotly_chart(fig_multi,use_container_width=True)
        st.dataframe(multi_df,use_container_width=True,hide_index=True)

        # Export
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
            # Full report
            s_pct_avail=all_pct3
            s_res_avail=all_res3
            rows_pct_all=[{"Canal":c,**{m:round(s_pct_avail[m][c],1) for m in model_names3}} for c in channels]
            rows_rev_all=[{"Canal":c,**{f"{m} ({curr_sym})":round(s_res_avail[m].get(c,0),0) for m in model_names3}} for c in channels]
            all_sheets={
                "KPIs":              (pd.DataFrame({"Métrica":["Journeys","Conversiones","Tasa conv. (%)","Revenue total","Ticket prom."],
                                                    "Valor":[n_total,n_conv,round(conv_rate,1),round(total_rev,0),round(avg_order,0)]}),
                                      {"Métrica":"text","Valor":"text"},"Resumen del dataset"),
                "Frecuencia canales":(pd.DataFrame({"Canal":list(ch_cnt.keys()),"Touchpoints":list(ch_cnt.values())}).sort_values("Touchpoints",ascending=False),
                                      {"Canal":"text","Touchpoints":"int"},"Touchpoints por canal"),
                "Crédito (%)":       (pd.DataFrame(rows_pct_all),{"Canal":"text",**{m:"pct" for m in model_names3}},"Crédito por canal (%)"),
                "Crédito rev.":      (pd.DataFrame(rows_rev_all),{"Canal":"text",**{f"{m} ({curr_sym})":"money" for m in model_names3}},f"Revenue atribuido ({curr_sym})"),
                "Escenario activo":  (bud_exp,bud_fmts,f"Escenario: {scenario_name}"),
                "Multi-modelo":      (multi_df,multi_fmts,f"Inversión por modelo"),
            }
            buf_all=build_excel(all_sheets)
            st.download_button("⬇️ Reporte completo (.xlsx)",buf_all,"attribution_reporte_completo.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",type="primary")
    except Exception as e:
        st.error(f"Error en escenarios: {e}")
        st.exception(e)
