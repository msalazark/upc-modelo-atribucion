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
.insight-card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
  padding:.65rem 1rem;margin:.3rem 0;font-size:.87rem;color:#334155;line-height:1.5}
.pill-blue{display:inline-block;background:#dbeafe;color:#1e40af;border-radius:20px;
  padding:2px 11px;font-size:.78rem;font-weight:600}
.pill-purple{display:inline-block;background:#ede9fe;color:#6d28d9;border-radius:20px;
  padding:2px 11px;font-size:.78rem;font-weight:600}
.rec-box{background:linear-gradient(135deg,#f0f9ff,#f0fdf4);border-left:4px solid #3b82f6;
  border-radius:8px;padding:1rem 1.2rem;font-size:.88rem;color:#1e293b;line-height:1.7}
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

# ── Simulator constants ───────────────────────────────────────────────────────
_CH_COLORS={"Meta Ads":"#3b82f6","Google Search":"#10b981","TikTok Ads":"#ec4899",
    "Display / Remarketing":"#f59e0b","Email / CRM":"#8b5cf6","SEO / Organic":"#06b6d4"}
_SATURATION_MULT={"Baja":1.00,"Media":0.85,"Alta":0.70}
_CAMPAIGN_TYPES =["Always On","Día D","Lanzamiento","Retención","Recorte"]
_OBJECTIVES_SIM =["Conversiones","Ingresos","ROAS","Utilidad bruta","Crecimiento"]
_ATT_MODELS_SIM =["Last Click","First Click","Lineal","Position-Based","Time Decay","Data-Driven Simulado"]
_SCENARIOS_SIM  =["Base","Performance","Awareness","Balanceado","Recorte","Día D"]
_FUNNEL_ROLES   =["Awareness","Consideración","Cierre","Retención","Awareness/Retención"]
_ATT_ROLE_W={
    "Last Click":    {"Awareness":0.05,"Consideración":0.15,"Cierre":1.00,"Retención":0.50,"Awareness/Retención":0.30},
    "First Click":   {"Awareness":1.00,"Consideración":0.20,"Cierre":0.10,"Retención":0.20,"Awareness/Retención":0.70},
    "Lineal":        {"Awareness":1.00,"Consideración":1.00,"Cierre":1.00,"Retención":1.00,"Awareness/Retención":1.00},
    "Position-Based":{"Awareness":0.80,"Consideración":0.30,"Cierre":0.80,"Retención":0.50,"Awareness/Retención":0.65},
    "Time Decay":    {"Awareness":0.20,"Consideración":0.45,"Cierre":1.00,"Retención":0.40,"Awareness/Retención":0.30},
    "Data-Driven Simulado":None,
}
_SCENARIO_ALLOC={
    "Base":       [0.278,0.333,0.111,0.089,0.056,0.133],
    "Performance":[0.200,0.420,0.080,0.120,0.060,0.120],
    "Awareness":  [0.350,0.150,0.250,0.050,0.050,0.150],
    "Balanceado": [0.250,0.250,0.150,0.100,0.100,0.150],
    "Recorte":    [0.200,0.380,0.040,0.040,0.140,0.200],
    "Día D":      [0.300,0.350,0.200,0.050,0.050,0.050],
}
_SCENARIO_DESC={
    "Base":       "Distribución de referencia sin cambios.",
    "Performance":"Concentra en Google Search y Display para maximizar conversiones directas.",
    "Awareness":  "Refuerza Meta y TikTok para descubrimiento de marca y nuevo alcance.",
    "Balanceado": "Reparte equitativamente entre etapas del funnel.",
    "Recorte":    "Eficiencia máxima: corta canales caros, mantiene rentables.",
    "Día D":      "Pauta masiva para un evento puntual: lanzamiento o fecha clave.",
}
_ATT_MODEL_DESC={
    "Last Click":          "100% del crédito al último canal antes de la conversión.",
    "First Click":         "100% del crédito al primer canal. Valora el descubrimiento.",
    "Lineal":              "Crédito repartido en partes iguales entre todos los canales.",
    "Position-Based":      "40% primer canal + 40% último canal + 20% entre intermedios.",
    "Time Decay":          "Más crédito a canales más cercanos en el tiempo a la conversión.",
    "Data-Driven Simulado":"Peso proporcional a adstock × confianza de tracking.",
}
_OBJECTIVE_TIPS={
    "Conversiones":  "Prioriza CPA bajo. Enfoca en canales con alta tasa de conversión y CPC razonable.",
    "Ingresos":      "Maximiza ticket × volumen. Evalúa canales con ticket promedio alto.",
    "ROAS":          "Maximiza ingresos por sol invertido. Corta canales con ROAS < 2×.",
    "Utilidad bruta":"El margen importa más que el revenue. Un buen ROAS puede tener margen bajo.",
    "Crecimiento":   "Invierte en awareness y canales con baja saturación. Piensa a mediano plazo.",
}
_DEFAULT_SIM=[
    {"canal":"Meta Ads",              "inv. actual":25000,"inv. simulada":25000,"CPC":0.85,"tasa conv.(%)":2.5,"ticket promedio":180,"margen(%)":42,"saturación":"Media","rol funnel":"Awareness/Retención","lag (días)":3, "adstock":1.15,"tracking(%)":70},
    {"canal":"Google Search",         "inv. actual":30000,"inv. simulada":30000,"CPC":1.20,"tasa conv.(%)":4.8,"ticket promedio":210,"margen(%)":40,"saturación":"Media","rol funnel":"Cierre",             "lag (días)":1, "adstock":1.05,"tracking(%)":90},
    {"canal":"TikTok Ads",            "inv. actual":10000,"inv. simulada":10000,"CPC":0.55,"tasa conv.(%)":1.2,"ticket promedio":150,"margen(%)":44,"saturación":"Baja", "rol funnel":"Awareness",           "lag (días)":5, "adstock":1.25,"tracking(%)":55},
    {"canal":"Display / Remarketing", "inv. actual":8000, "inv. simulada":8000, "CPC":0.40,"tasa conv.(%)":2.0,"ticket promedio":195,"margen(%)":41,"saturación":"Alta", "rol funnel":"Consideración",       "lag (días)":2, "adstock":1.10,"tracking(%)":75},
    {"canal":"Email / CRM",           "inv. actual":5000, "inv. simulada":5000, "CPC":0.10,"tasa conv.(%)":6.5,"ticket promedio":220,"margen(%)":48,"saturación":"Baja", "rol funnel":"Retención",           "lag (días)":1, "adstock":1.00,"tracking(%)":85},
    {"canal":"SEO / Organic",         "inv. actual":12000,"inv. simulada":12000,"CPC":0.25,"tasa conv.(%)":3.2,"ticket promedio":175,"margen(%)":46,"saturación":"Baja", "rol funnel":"Awareness",           "lag (días)":14,"adstock":1.30,"tracking(%)":60},
]

# ── Simulator functions ────────────────────────────────────────────────────────
def safe_div(a,b,default=0.0):
    return a/b if b and not(isinstance(b,float) and np.isnan(b)) else default

def sim_apply_scenario(df,budget,scenario):
    df=df.copy()
    ratios=_SCENARIO_ALLOC.get(scenario,_SCENARIO_ALLOC["Base"])
    for i in range(len(df)):
        r=ratios[i] if i<len(ratios) else 1/len(df)
        df.at[i,"inv. simulada"]=round(budget*r)
    return df

def sim_calc_metrics(df):
    rows=[]
    for _,r in df.iterrows():
        inv    =max(float(r["inv. simulada"]),0)
        cpc    =max(float(r["CPC"]),0.001)
        tcr    =float(r["tasa conv.(%)"]) /100
        tp     =max(float(r["ticket promedio"]),0)
        mb     =float(r["margen(%)"])/100
        sat_m  =_SATURATION_MULT.get(str(r["saturación"]),1.0)
        adstock=max(float(r["adstock"]),0.01)
        clicks =safe_div(inv,cpc)
        convs  =clicks*tcr*sat_m*adstock
        rev    =convs*tp; margin=rev*mb
        rows.append({"canal":r["canal"],"inversión":inv,"clicks":round(clicks),
            "conversiones":round(convs,1),"ingresos":round(rev,2),
            "utilidad bruta":round(margin,2),
            "CPA":round(safe_div(inv,convs),2),"ROAS":round(safe_div(rev,inv),2)})
    return pd.DataFrame(rows)

def sim_calc_base(df):
    d=df.copy(); d["inv. simulada"]=d["inv. actual"]; return sim_calc_metrics(d)

def sim_calc_attribution(df_m,df_in,model):
    if model=="Data-Driven Simulado":
        w=df_in["adstock"].values*(df_in["tracking(%)"].values/100)
    else:
        w=df_in["rol funnel"].map(_ATT_ROLE_W[model]).fillna(1.0).values
    raw_w=w*df_m["conversiones"].values; tot_w=raw_w.sum()
    norm=raw_w/tot_w if tot_w>0 else np.ones(len(df_m))/max(len(df_m),1)
    res=df_m[["canal"]].copy()
    res["conv. atribuidas"]    =(norm*df_m["conversiones"].sum()).round(1)
    res["ingresos atribuidos"] =(norm*df_m["ingresos"].sum()).round(2)
    res["crédito (%)"]         =(norm*100).round(1)
    return res.reset_index(drop=True)

def _sim_colors(chs):
    return [_CH_COLORS.get(c,"#94a3b8") for c in chs]

def _sim_layout(fig,title,ytitle,height=270):
    fig.update_layout(title_text=title,height=height,margin=dict(l=0,r=0,t=36,b=8),
        yaxis_title=ytitle,plot_bgcolor="white",paper_bgcolor="white",
        legend=dict(orientation="h",y=1.14))

def sim_chart_inversion(df):
    chs=df["canal"].tolist()
    fig=go.Figure([
        go.Bar(name="Actual",  x=chs,y=df["inv. actual"].tolist(),  marker_color="#cbd5e1",opacity=0.85),
        go.Bar(name="Simulada",x=chs,y=df["inv. simulada"].tolist(),marker_color=_sim_colors(chs),
               text=[f"{v:,.0f}" for v in df["inv. simulada"]],textposition="outside"),
    ])
    fig.update_layout(barmode="group"); _sim_layout(fig,"Inversión por canal","S/"); return fig

def sim_chart_conversiones(dm,db):
    chs=dm["canal"].tolist()
    fig=go.Figure([
        go.Bar(name="Base",    x=chs,y=db["conversiones"].tolist(),marker_color="#cbd5e1",opacity=0.85),
        go.Bar(name="Simulado",x=chs,y=dm["conversiones"].tolist(),marker_color=_sim_colors(chs),
               text=[f"{v:.0f}" for v in dm["conversiones"]],textposition="outside"),
    ])
    fig.update_layout(barmode="group"); _sim_layout(fig,"Conversiones estimadas","Conv."); return fig

def sim_chart_roas(dm,db):
    chs=dm["canal"].tolist()
    fig=go.Figure([
        go.Bar(name="Base",    x=chs,y=db["ROAS"].tolist(),marker_color="#cbd5e1",opacity=0.85),
        go.Bar(name="Simulado",x=chs,y=dm["ROAS"].tolist(),marker_color=_sim_colors(chs),
               text=[f"{v:.1f}×" for v in dm["ROAS"]],textposition="outside"),
    ])
    fig.update_layout(barmode="group")
    fig.add_hline(y=2,line_dash="dash",line_color="#ef4444",opacity=0.5,
                  annotation_text="Mínimo 2×",annotation_position="bottom right")
    _sim_layout(fig,"ROAS por canal","ROAS"); return fig

def sim_chart_cpa(dm,db):
    chs=dm["canal"].tolist()
    vc=dm["CPA"].replace(0,np.nan); bc=db["CPA"].replace(0,np.nan)
    fig=go.Figure([
        go.Bar(name="Base",    x=chs,y=bc.tolist(),marker_color="#cbd5e1",opacity=0.85),
        go.Bar(name="Simulado",x=chs,y=vc.tolist(),marker_color=_sim_colors(chs),
               text=[f"{v:,.0f}" if not np.isnan(v) else "—" for v in vc],textposition="outside"),
    ])
    fig.update_layout(barmode="group"); _sim_layout(fig,"CPA por canal (S/)","S/"); return fig

def sim_chart_credito(da,model):
    chs=da["canal"].tolist()
    fig=go.Figure(go.Bar(x=chs,y=da["crédito (%)"].tolist(),marker_color=_sim_colors(chs),
        text=[f"{v:.1f}%" for v in da["crédito (%)"]],textposition="outside"))
    _sim_layout(fig,f"Crédito atribuido — {model}","Crédito (%)")
    fig.update_layout(yaxis_ticksuffix="%"); return fig

def sim_chart_ingresos_att(da,dm):
    chs=da["canal"].tolist()
    fig=go.Figure([
        go.Bar(name="Generados", x=chs,y=dm["ingresos"].tolist(),         marker_color="#cbd5e1",opacity=0.85),
        go.Bar(name="Atribuidos",x=chs,y=da["ingresos atribuidos"].tolist(),marker_color=_sim_colors(chs),
               text=[f"{v:,.0f}" for v in da["ingresos atribuidos"]],textposition="outside"),
    ])
    fig.update_layout(barmode="group"); _sim_layout(fig,"Ingresos atribuidos vs generados","S/"); return fig

def sim_generate_insights(dm,di,objective,att_model):
    ins=[]
    # ROAS – explica qué significa el número
    pos=dm[dm["ROAS"]>0]
    if not pos.empty:
        b=pos.loc[pos["ROAS"].idxmax()]; rv=b["ROAS"]
        nivel="excelente" if rv>=4 else ("aceptable" if rv>=2 else "bajo")
        consejo=("Hay margen para aumentar inversión si la saturación es baja."
                 if rv>=2 else "Revisa CPC, tasa de conversión o ticket promedio.")
        ins.append(("📊",
            f"<b>{b['canal']}</b> tiene el mejor ROAS: <b>{rv:.1f}×</b> ({nivel}). "
            f"Eso significa que por cada S/1 invertido aquí se generan S/{rv:.1f} en ventas. {consejo}"))
    # CPA – compara el más caro vs el más barato
    pc=dm[dm["CPA"]>0]
    if len(pc)>=2:
        wc=pc.loc[pc["CPA"].idxmax()]; bc=pc.loc[pc["CPA"].idxmin()]
        ins.append(("💰",
            f"El costo por conversión varía mucho entre canales: "
            f"<b>{bc['canal']}</b> convierte a S/ <b>{bc['CPA']:,.0f}</b>, "
            f"mientras que <b>{wc['canal']}</b> cuesta S/ <b>{wc['CPA']:,.0f}</b>. "
            f"Si el ticket promedio es similar, mover presupuesto al canal más barato mejora la eficiencia."))
    # Saturación – explica la consecuencia práctica
    hi=di[di["saturación"]=="Alta"]["canal"].tolist()
    if hi:
        ins.append(("🔴",
            f"<b>{', '.join(hi)}</b>: saturación alta. La audiencia ya vio muchas veces el anuncio. "
            f"Agregar más presupuesto rinde ~30% menos que en condiciones normales. "
            f"Opciones: renovar la creatividad, ampliar la audiencia o redistribuir a otro canal."))
    # Tracking – desmitifica el dato bajo
    lt=di[di["tracking(%)"]<65]["canal"].tolist()
    if lt:
        ins.append(("🔍",
            f"<b>{', '.join(lt)}</b>: confianza de tracking baja. "
            f"Esto no significa que el canal funcione mal — significa que no estamos midiendo bien. "
            f"Con UTMs y píxel bien configurados, probablemente veríamos más conversiones atribuidas."))
    # Modelo de atribución – ejemplo concreto
    att_ctx={
        "Last Click":   "Con <b>Last Click</b>: si un cliente vio un anuncio en Meta, luego buscó en Google y compró, Google recibe el 100% del crédito. Meta queda con 0%, aunque fue quien generó el interés inicial.",
        "First Click":  "Con <b>First Click</b>: el canal que inicia el journey recibe todo el crédito. Útil para medir qué canal descubre clientes nuevos, pero ignora qué canal cerró la venta.",
        "Lineal":       "Con <b>Lineal</b>: si un cliente tocó 4 canales, cada uno recibe 25% del crédito. Ningún canal domina artificialmente — el modelo es neutro.",
        "Position-Based":"Con <b>Position-Based</b>: el primer canal recibe 40%, el último canal 40%, y el resto comparte el 20%. Reconoce tanto el descubrimiento como el cierre.",
        "Time Decay":   "Con <b>Time Decay</b>: el canal más cercano a la compra recibe más crédito. Si Meta actuó hace 14 días y Google hace 1 día, Google recibe mucho más crédito.",
        "Data-Driven Simulado":"Con <b>Data-Driven</b>: el crédito se pondera por adstock × confianza de tracking. Canales con mayor efecto acumulado y mejor medición capturan más crédito.",
    }
    if att_model in att_ctx:
        ins.append(("📌", att_ctx[att_model]))
    # CRM subinvertido – explica el por qué
    crm=di[di["canal"].str.contains("Email|CRM",case=False)]
    if not crm.empty:
        cp=safe_div(crm["inv. simulada"].sum(),di["inv. simulada"].sum())*100
        tcr=crm["tasa conv.(%)"].mean()
        if cp<8 and tcr>4:
            ins.append(("💡",
                f"<b>Email/CRM</b> tiene {tcr:.1f}% de tasa de conversión — una de las más altas — "
                f"pero solo recibe el {cp:.1f}% del presupuesto. Es tu base de clientes actuales: "
                f"ya confían en ti y ya conocen el producto. Suele ser el canal con menor CPA."))
    # Awareness para objetivo crecimiento
    if objective=="Crecimiento":
        aw=di[di["rol funnel"].isin(["Awareness","Awareness/Retención"])]["inv. simulada"].sum()
        ap=safe_div(aw,di["inv. simulada"].sum())*100
        if ap<25:
            ins.append(("💡",
                f"Solo el {ap:.0f}% del presupuesto va a canales de <b>Awareness</b>. "
                f"Para un objetivo de Crecimiento necesitas nuevos clientes, no solo reactivar existentes. "
                f"Se recomienda destinar al menos 25–35% a descubrimiento de marca."))
    # Lag – explica cómo evaluarlos
    lag_hi=di[di["lag (días)"]>7]["canal"].tolist()
    if lag_hi:
        ins.append(("⏳",
            f"<b>{', '.join(lag_hi)}</b>: lag mayor a 7 días. Esto significa que si inviertes hoy, "
            f"los resultados aparecen en semanas, no en días. "
            f"No los juzgues con reportes semanales — compara períodos de 30–60 días."))
    return ins

def sim_generate_recommendation(dm,di,da,objective,att_model,campaign_type,scenario,sym):
    ti=di["inv. simulada"].sum(); tc=dm["conversiones"].sum()
    tr=dm["ingresos"].sum();      tm=dm["utilidad bruta"].sum()
    roas=safe_div(tr,ti);         cpa=safe_div(ti,tc)
    best=dm.loc[dm["ROAS"].idxmax(),"canal"] if not dm.empty else "—"
    top_a=da.loc[da["crédito (%)"].idxmax(),"canal"] if not da.empty else "—"
    hi_s=di[di["saturación"]=="Alta"]["canal"].tolist()
    roas_txt="sólido (≥3×)" if roas>=3 else ("aceptable (2–3×)" if roas>=2 else "bajo (<2×, revisar)")
    obj_adv={
        "Conversiones":  f"Concentra incrementos en **{best}** (mejor ROAS) y revisa Email/CRM si no está saturado.",
        "Ingresos":      f"Prioriza canales con ticket alto y tasa de conversión sólida. Utilidad bruta estimada: {sym} {tm:,.0f}.",
        "ROAS":          f"ROAS {roas:.1f}× es {roas_txt}. {'Corta canales con ROAS < 1.5× y redistribuye a '+best+'.' if roas<2 else 'Mantén la mezcla y optimiza creatividades.'}",
        "Utilidad bruta":f"Prioriza canales con margen > 45%. Un ROAS alto con margen bajo puede no justificar la inversión.",
        "Crecimiento":   "Refuerza TikTok y SEO (bajo CPC, baja saturación, adstock alto). El impacto se ve en 2–4 semanas.",
    }
    parts=[
        f"**Escenario {scenario}** · Campaña: {campaign_type} · Objetivo: {objective} · Modelo: {att_model}",
        "",
        f"Con **{sym} {ti:,.0f}** de inversión simulada, el modelo estima **{tc:,.0f} conversiones** "
        f"y **{sym} {tr:,.0f}** en ingresos. ROAS: **{roas:.1f}×** ({roas_txt}). "
        f"CPA promedio: **{sym} {cpa:,.0f}**.",
        "",
        obj_adv.get(objective,""),
    ]
    if att_model=="Last Click":
        parts.append(f"\n⚠️ **Last Click** asigna el mayor crédito a **{top_a}**, pero puede ignorar "
            "el trabajo de canales de awareness. Valida con un modelo multi-touch antes de cortes.")
    elif att_model=="Data-Driven Simulado":
        parts.append("\nEl modelo **Data-Driven** pondera adstock × confianza de tracking. "
            "Mejora el tagging de canales con tracking bajo para obtener pesos más precisos.")
    else:
        parts.append(f"\n**{att_model}** distribuye el crédito de forma más equitativa, "
            "reduciendo el riesgo de subestimar canales de funnel medio.")
    if hi_s:
        parts.append(f"\n🔴 **{', '.join(hi_s)}** con saturación alta: el presupuesto marginal "
            "aquí rinde ~30% menos. Redirige ese excedente a canales con capacidad de escala.")
    return "\n\n".join(parts)

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
                    val=df_s.iloc[ri,ci]
                    fmt=alt[fk] if a else base[fk]
                    if pd.isna(val):
                        ws.write_blank(ri+3,ci,None,fmt)
                    else:
                        ws.write(ri+3,ci,val,fmt)
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
        # ── Config bar (inline) ───────────────────────────────────────────────
        sim_budget = budget + seo_budget
        cfg1,cfg2,cfg3,cfg4 = st.columns(4)
        with cfg1:
            st.markdown("**Tipo de campaña**")
            campaign_type=st.selectbox("",_CAMPAIGN_TYPES,label_visibility="collapsed",key="sim_camp")
        with cfg2:
            st.markdown("**Objetivo principal**")
            objective_sim=st.selectbox("",_OBJECTIVES_SIM,label_visibility="collapsed",key="sim_obj")
        with cfg3:
            st.markdown("**Modelo de atribución**")
            att_model_sim=st.selectbox("",_ATT_MODELS_SIM,label_visibility="collapsed",key="sim_att")
        with cfg4:
            st.markdown("**Escenario**")
            scenario_sim=st.selectbox("",_SCENARIOS_SIM,label_visibility="collapsed",key="sim_sc")

        st.markdown(
            f'<span class="pill-blue">Escenario: {scenario_sim}</span>&nbsp;'
            f'<span class="pill-purple">Modelo: {att_model_sim}</span>&nbsp;&nbsp;'
            f'<span style="font-size:.82rem;color:#64748b;">{_SCENARIO_DESC[scenario_sim]}</span>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="info-box" style="font-size:.8rem;margin-top:.3rem;">'
            f'<b>Objetivo {objective_sim}:</b> {_OBJECTIVE_TIPS[objective_sim]} &mdash; '
            f'<b>Atribución:</b> {_ATT_MODEL_DESC[att_model_sim]}</div>',
            unsafe_allow_html=True)

        # ── Session state & scenario reset ────────────────────────────────────
        if "sim_df" not in st.session_state:
            st.session_state["sim_df"] = pd.DataFrame(_DEFAULT_SIM)

        cfg_key = f"{scenario_sim}_{sim_budget}"
        if st.session_state.get("_sim_cfg") != cfg_key:
            st.session_state["_sim_cfg"] = cfg_key
            st.session_state["sim_df"] = sim_apply_scenario(
                st.session_state["sim_df"], sim_budget, scenario_sim)

        # ── Data editor ───────────────────────────────────────────────────────
        st.markdown('<p class="section-title">📋 Tabla de canales — edita para simular</p>',
            unsafe_allow_html=True)

        with st.expander("📖 ¿Qué significa cada columna? Haz clic para ver la guía"):
            st.markdown("""
| Columna | Qué representa | Ejemplo concreto |
|---|---|---|
| **Inv. simulada** | Cuánto dinero asignas a este canal | S/ 30,000 al mes |
| **CPC** | Costo por clic — cuánto pagas cada vez que alguien hace clic en tu anuncio | S/ 1.20 por clic |
| **Conv. (%)** | De cada 100 personas que hacen clic, cuántas terminan comprando | 4.8% → ~5 de cada 100 |
| **Ticket** | Cuánto vale en promedio cada venta | S/ 210 por pedido |
| **Margen (%)** | De cada venta, qué porcentaje queda como ganancia después de pagar el costo del producto | 40% → ganas S/ 84 de una venta de S/ 210 |
| **Saturación** | Si ya mostraste muchos anuncios a la misma audiencia | Alta → la audiencia ya está "fatigada" del anuncio |
| **Rol funnel** | En qué etapa del proceso de compra actúa este canal | Awareness = descubrimiento · Cierre = última acción antes de comprar |
| **Lag** | Cuántos días pasan en promedio entre que alguien ve el anuncio y compra | 14 días para SEO → el efecto es lento pero duradero |
| **Adstock** | Si la publicidad acumula efecto en el tiempo (más allá del día que se invirtió) | 1.30 → 30% del impacto se extiende más allá del período |
| **Tracking (%)** | Qué tan confiables son los datos de conversión de este canal | 55% → casi la mitad de las conversiones podrían no estar siendo contadas |
""")
            st.markdown("**Pista:** empieza cambiando la *Inv. simulada* de un canal y observa cómo cambian el ROAS y el CPA en los indicadores de abajo.")

        st.markdown(
            '<div style="background:#fef9c3;border:2px dashed #eab308;border-radius:8px;'
            'padding:.7rem 1rem;font-size:.9rem;color:#713f12;margin-bottom:.5rem;">'
            '✏️ <b>La tabla de abajo es editable.</b> Haz clic en cualquier celda de color blanco '
            'para cambiar el valor. Las columnas <b>Canal</b> e <b>Inv. actual</b> son de referencia '
            '(no se pueden editar). Empieza por cambiar la columna <b>Inv. simulada</b>.</div>',
            unsafe_allow_html=True)

        col_cfg={
            "canal":          st.column_config.TextColumn("Canal",disabled=True,width="medium"),
            "inv. actual":    st.column_config.NumberColumn(f"Inv. actual ({curr_sym})",disabled=True,
                              format=f"{curr_sym} %,.0f"),
            "inv. simulada":  st.column_config.NumberColumn(f"Inv. simulada ({curr_sym})",min_value=0,
                              format=f"{curr_sym} %,.0f",
                              help="Modifica para simular el nuevo escenario."),
            "CPC":            st.column_config.NumberColumn(f"CPC ({curr_sym})",min_value=0.01,
                              max_value=50.0,format=f"{curr_sym} %.2f",
                              help="Costo por clic = inversión ÷ clicks."),
            "tasa conv.(%)":  st.column_config.NumberColumn("Conv.(%)",min_value=0.0,max_value=100.0,
                              step=0.1,format="%.1f %%",
                              help="% de clics que se convierten."),
            "ticket promedio":st.column_config.NumberColumn(f"Ticket ({curr_sym})",min_value=0,
                              format=f"{curr_sym} %,.0f",
                              help="Valor promedio por conversión."),
            "margen(%)":      st.column_config.NumberColumn("Margen(%)",min_value=0,max_value=100,
                              step=1,format="%.0f %%",
                              help="% de ingresos como utilidad bruta."),
            "saturación":     st.column_config.SelectboxColumn("Saturación",
                              options=["Baja","Media","Alta"],
                              help="Baja: sin penalización · Media: −15% · Alta: −30% eficiencia."),
            "rol funnel":     st.column_config.SelectboxColumn("Rol funnel",options=_FUNNEL_ROLES,
                              help="Define el peso del canal en el modelo de atribución."),
            "lag (días)":     st.column_config.NumberColumn("Lag (días)",min_value=0,max_value=60,
                              help="Días de retraso entre inversión y conversión observable."),
            "adstock":        st.column_config.NumberColumn("Adstock",min_value=0.5,max_value=2.5,
                              step=0.05,format="%.2f",
                              help="Multiplicador de carryover: >1 indica que la pauta acumula efecto."),
            "tracking(%)":    st.column_config.NumberColumn("Tracking(%)",min_value=0,max_value=100,
                              step=5,format="%.0f %%",
                              help="Confianza en la medición de conversiones. Bajo = datos incompletos."),
        }
        edited=st.data_editor(st.session_state["sim_df"],column_config=col_cfg,
            use_container_width=True,hide_index=True,num_rows="fixed",key="t3_editor")
        st.session_state["sim_df"]=edited

        # ── Compute ───────────────────────────────────────────────────────────
        dm=sim_calc_metrics(edited)
        db=sim_calc_base(edited)
        da=sim_calc_attribution(dm,edited,att_model_sim)
        ti=float(edited["inv. simulada"].sum()); ti_b=float(edited["inv. actual"].sum())
        tc=dm["conversiones"].sum();             tc_b=db["conversiones"].sum()
        tr=dm["ingresos"].sum();                 tr_b=db["ingresos"].sum()
        tm=dm["utilidad bruta"].sum()
        cpa=safe_div(ti,tc); cpa_b=safe_div(ti_b,tc_b)
        roas=safe_div(tr,ti); roas_b=safe_div(tr_b,ti_b)

        # ── KPI cards ─────────────────────────────────────────────────────────
        st.divider()
        st.markdown('<p class="section-title">📌 Resultados proyectados del escenario</p>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box" style="font-size:.82rem;">Las flechas (↑↓) comparan el escenario simulado '
            'contra la inversión actual (base). Verde = mejora, rojo = empeora. '
            'En CPA, rojo significa que cuesta más conseguir una conversión — menor es mejor.</div>',
            unsafe_allow_html=True)
        k1,k2,k3,k4,k5,k6=st.columns(6)
        k1.metric(f"Inversión ({curr_sym})", f"{ti:,.0f}",   delta=f"{ti-ti_b:+,.0f}")
        k2.metric("Conversiones",            f"{tc:,.0f}",   delta=f"{tc-tc_b:+,.0f}")
        k3.metric(f"Ingresos ({curr_sym})",  f"{tr:,.0f}",   delta=f"{tr-tr_b:+,.0f}")
        k4.metric(f"Utilidad bruta ({curr_sym})", f"{tm:,.0f}", help="Ingresos × margen bruto ponderado. Es lo que queda antes de sueldos y gastos fijos.")
        k5.metric(f"CPA ({curr_sym})",       f"{cpa:,.0f}",  delta=f"{cpa-cpa_b:+,.0f}",delta_color="inverse")
        k6.metric("ROAS total",              f"{roas:.2f}×", delta=f"{roas-roas_b:+.2f}×")

        # Contexto ROAS y CPA
        roas_color="#16a34a" if roas>=3 else ("#d97706" if roas>=2 else "#dc2626")
        roas_label="Excelente ✓" if roas>=3 else ("Aceptable" if roas>=2 else "Bajo — revisa los supuestos")
        st.markdown(
            f'<div style="display:flex;gap:.8rem;margin:.5rem 0 0;">'
            f'<div class="info-box" style="flex:1;margin:0;">'
            f'<b>¿Qué es el ROAS?</b> Return On Ad Spend = ingresos ÷ inversión.<br>'
            f'Tu ROAS de <b style="color:{roas_color};">{roas:.1f}× ({roas_label})</b> significa que '
            f'por cada {curr_sym} 1 invertido, generas {curr_sym} {roas:.1f} en ventas.<br>'
            f'<span style="font-size:.8rem;color:#64748b;">Referencia: &lt;2× bajo · 2–3× aceptable · &gt;3× excelente</span></div>'
            f'<div class="info-box" style="flex:1;margin:0;">'
            f'<b>¿Qué es el CPA?</b> Costo Por Adquisición = inversión ÷ conversiones.<br>'
            f'Tu CPA promedio de <b>{curr_sym} {cpa:,.0f}</b> es lo que cuesta conseguir una venta o lead.<br>'
            f'<span style="font-size:.8rem;color:#64748b;">Regla rápida: CPA debe ser menor que el ticket promedio × margen.</span></div>'
            f'</div>',
            unsafe_allow_html=True)

        # ── Charts ────────────────────────────────────────────────────────────
        st.divider()
        st.markdown('<p class="section-title">📊 Comparación por canal: base vs simulado</p>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box" style="font-size:.82rem;">'
            '<b>Cómo leer los gráficos:</b> las barras grises son el escenario base (inversión actual). '
            'Las barras de color son el escenario simulado. '
            'Si la barra de color es más alta → ese canal mejora en el escenario nuevo. '
            'Si son iguales → no cambió.</div>',
            unsafe_allow_html=True)
        r1,r2=st.columns(2)
        with r1: st.plotly_chart(sim_chart_inversion(edited),    use_container_width=True)
        with r2: st.plotly_chart(sim_chart_conversiones(dm,db),  use_container_width=True)

        st.markdown(
            '<div class="info-box" style="font-size:.82rem;">'
            '<b>Inversión ≠ Conversiones:</b> un canal puede recibir más presupuesto y generar menos conversiones '
            'si tiene CPC alto, baja tasa de conversión o saturación alta. '
            'Compara siempre ambos gráficos juntos.</div>',
            unsafe_allow_html=True)

        r3,r4=st.columns(2)
        with r3: st.plotly_chart(sim_chart_roas(dm,db),          use_container_width=True)
        with r4: st.plotly_chart(sim_chart_cpa(dm,db),           use_container_width=True)
        st.markdown(
            '<div class="info-box" style="font-size:.82rem;">'
            '<b>ROAS y CPA miden eficiencia, no volumen.</b> '
            'Un canal puede tener excelente ROAS pero pocas conversiones (canal pequeño pero rentable). '
            'La línea roja punteada en ROAS marca el mínimo de 2× — por debajo, el canal apenas cubre costos.</div>',
            unsafe_allow_html=True)

        # ── Attribution ───────────────────────────────────────────────────────
        st.divider()
        st.markdown(f'<p class="section-title">🏅 ¿A quién le damos el crédito? — Modelo: {att_model_sim}</p>',
            unsafe_allow_html=True)

        _att_examples={
            "Last Click":   ("Ana ve un anuncio en Meta → busca en Google → compra.",
                             "Google recibe el 100% del crédito. Meta recibe 0%, aunque fue el primer contacto.",
                             "Útil para optimizar el canal de cierre. Riesgo: infravalora el awareness."),
            "First Click":  ("Ana ve un anuncio en TikTok → recibe un email → compra.",
                             "TikTok recibe el 100% del crédito. El email que cerró la venta recibe 0%.",
                             "Útil para medir qué canal descubre nuevos clientes."),
            "Lineal":       ("Ana toca Meta, TikTok, Google y Email antes de comprar.",
                             "Cada canal recibe exactamente el 25% del crédito (4 canales = 25% c/u).",
                             "Neutral y simple. No asume que ningún canal importa más que otro."),
            "Position-Based":("Ana toca Meta → TikTok → Google → Email → compra.",
                              "Meta: 40% · Email: 40% · TikTok y Google: 10% c/u.",
                              "Valora tanto el inicio (descubrimiento) como el cierre de la venta."),
            "Time Decay":   ("Ana ve Meta hace 14 días, luego Google hace 2 días, luego compra.",
                             "Google recibe mucho más crédito por estar más cerca en el tiempo.",
                             "Útil en ciclos de compra cortos. Penaliza canales de awareness de largo plazo."),
            "Data-Driven Simulado":("Se pondera por adstock × confianza de tracking de cada canal.",
                                    "Canales con mayor efecto acumulado y mejor medición capturan más crédito.",
                                    "Más realista si los datos de tracking son confiables (>70%)."),
        }
        ex_sit,ex_res,ex_tip=_att_examples.get(att_model_sim,("","",""))

        al,ar=st.columns([3,2])
        with al:
            ac1,ac2=st.columns(2)
            with ac1: st.plotly_chart(sim_chart_credito(da,att_model_sim),  use_container_width=True)
            with ac2: st.plotly_chart(sim_chart_ingresos_att(da,dm),        use_container_width=True)
        with ar:
            st.markdown(
                f'<div class="info-box">'
                f'<b>Ejemplo con {att_model_sim}:</b><br><br>'
                f'🧍 <b>Situación:</b> {ex_sit}<br><br>'
                f'📊 <b>Resultado:</b> {ex_res}<br><br>'
                f'💡 <b>Cuándo usarlo:</b> {ex_tip}</div>',
                unsafe_allow_html=True)
            st.markdown('<p class="section-title">Crédito por canal</p>',unsafe_allow_html=True)
            att_show=da.rename(columns={"canal":"Canal","crédito (%)":"Crédito (%)",
                "conv. atribuidas":"Conv. atribuidas",
                "ingresos atribuidos":f"Ingresos ({curr_sym})"})
            st.dataframe(att_show,use_container_width=True,hide_index=True)
            st.markdown(
                '<div class="info-box" style="font-size:.8rem;margin-top:.5rem;">'
                '💡 Cambia el modelo en la barra de configuración y observa '
                'cómo el mismo escenario produce un reparto de crédito muy diferente.</div>',
                unsafe_allow_html=True)

        # ── Results table ─────────────────────────────────────────────────────
        st.divider()
        st.markdown('<p class="section-title">📄 Resultados por canal</p>',
            unsafe_allow_html=True)
        res=dm.rename(columns={"canal":"Canal","inversión":f"Inversión ({curr_sym})",
            "clicks":"Clicks","conversiones":"Conversiones","ingresos":f"Ingresos ({curr_sym})",
            "utilidad bruta":f"Utilidad bruta ({curr_sym})","CPA":f"CPA ({curr_sym})","ROAS":"ROAS"})
        st.dataframe(res.style.format({
            f"Inversión ({curr_sym})":"{:,.0f}","Clicks":"{:,.0f}","Conversiones":"{:.1f}",
            f"Ingresos ({curr_sym})":"{:,.0f}",f"Utilidad bruta ({curr_sym})":"{:,.0f}",
            f"CPA ({curr_sym})":"{:,.0f}","ROAS":"{:.2f}×"}),
            use_container_width=True,hide_index=True)

        with st.expander("📖 ¿Cómo leer esta tabla?"):
            dm_pos=dm[dm["CPA"]>0]
            best_roas_ch=dm.loc[dm["ROAS"].idxmax(),"canal"] if not dm.empty else "—"
            best_roas_v=dm["ROAS"].max()
            best_cpa_ch=dm_pos.loc[dm_pos["CPA"].idxmin(),"canal"] if not dm_pos.empty else "—"
            best_cpa_v=dm_pos["CPA"].min() if not dm_pos.empty else 0
            st.markdown(f"""
- **Clicks** = Inversión ÷ CPC. Si inviertes S/ 30,000 con CPC S/ 1.20 → consigues ~25,000 clics.
- **Conversiones** = Clicks × Tasa de conversión × Saturación × Adstock. Es el volumen de ventas proyectado.
- **Ingresos** = Conversiones × Ticket promedio. El dinero que entra por ventas.
- **Utilidad bruta** = Ingresos × Margen. Lo que queda después de pagar el costo del producto.
- **CPA** = Inversión ÷ Conversiones. Cuánto cuesta conseguir una venta. **Menor es mejor.**
- **ROAS** = Ingresos ÷ Inversión. Cuánto generas por cada sol invertido. **Mayor es mejor.**

En este escenario, **{best_roas_ch}** tiene el mejor ROAS ({best_roas_v:.1f}×) y **{best_cpa_ch}** tiene el CPA más bajo ({curr_sym} {best_cpa_v:,.0f}).
""")

        # ── Insights educativos ────────────────────────────────────────────────
        st.divider()
        st.markdown('<p class="section-title">💡 ¿Qué te dicen los datos?</p>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box" style="font-size:.82rem;">Cada observación explica el concepto detrás del dato '
            'y qué podrías hacer al respecto. Cambia parámetros en la tabla y observa cómo cambian.</div>',
            unsafe_allow_html=True)
        insights=sim_generate_insights(dm,edited,objective_sim,att_model_sim)
        for icon,text in insights:
            st.markdown(f'<div class="insight-card">{icon}&nbsp; {text}</div>',
                unsafe_allow_html=True)

        # ── Resumen simple ────────────────────────────────────────────────────
        st.divider()
        st.markdown('<p class="section-title">📝 Resumen del escenario</p>',
            unsafe_allow_html=True)

        roas_sem="🟢 excelente" if roas>=4 else ("🟡 aceptable" if roas>=2 else "🔴 bajo")
        best_ch=dm.loc[dm["ROAS"].idxmax(),"canal"] if not dm.empty else "—"
        best_v=dm["ROAS"].max()
        hi_sat=edited[edited["saturación"]=="Alta"]["canal"].tolist()
        low_tr=edited[edited["tracking(%)"]<65]["canal"].tolist()

        st.markdown(f"""
- **Eficiencia global:** ROAS {roas:.1f}× — {roas_sem}. Por cada {curr_sym} 1 invertido, se generan {curr_sym} {roas:.1f} en ventas.
- **Canal más eficiente:** {best_ch} con ROAS {best_v:.1f}×. Si su saturación es baja, hay espacio para escalar.
- **Objetivo del escenario ({objective_sim}):** {_OBJECTIVE_TIPS[objective_sim]}
""")
        if hi_sat:
            st.markdown(f"- **Saturación alta en {', '.join(hi_sat)}:** aumentar más presupuesto aquí rinde menos. Renueva la creatividad o redistribuye a otro canal.")
        if low_tr:
            st.markdown(f"- **Tracking limitado en {', '.join(low_tr)}:** los resultados reales podrían ser mejores que lo simulado. Configura bien el píxel y los UTMs.")
        st.markdown(f"- **¿Qué probar a continuación?** Cambia el modelo de atribución y compara cómo se redistribuye el crédito. Prueba el escenario **Performance** vs **Awareness** para ver qué estrategia funciona mejor con tu objetivo.")
        st.markdown(
            '<div class="info-box" style="margin-top:.6rem;">'
            '📌 <b>Recuerda:</b> este simulador usa supuestos simplificados. '
            'Los valores reales dependen de la calidad de las campañas, el mercado y la época del año. '
            'Úsalo como punto de partida para el análisis, no como predicción exacta.</div>',
            unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error en escenarios de inversión: {e}")
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
