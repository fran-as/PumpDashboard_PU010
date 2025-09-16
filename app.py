import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import streamlit as st
import plotly.express as px

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="PU-010 • Análisis Bomba vs Nivel",
    layout="wide",
)

DATA_PATH = Path("Data/dataset.xlsx")

# Columnas esperadas en el Excel
COL_TIME    = "Fecha"
COL_PUMPRPM = "VelocidadPU010_rpm"           # [rpm]
COL_LEVELP  = "NivelCajonHP010_Percent"      # [%]

# Parámetros de negocio
PUMP_MAX_RPM = 356.0  # techo físico de la bomba [rpm]
CUTOFF_DATE  = pd.Timestamp("2025-08-23")  # excluir >= esta fecha si usuario activa el botón

# Paleta consistente (bomba vs cajón)
COLOR_PUMP  = "#1f77b4"  # azul
COLOR_LEVEL = "#d62728"  # rojo

# =========================
# UTILIDADES
# =========================
@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    """Carga dataset Excel y tipifica columnas."""
    df = pd.read_excel(path)
    # Validación
    expected = [COL_TIME, COL_PUMPRPM, COL_LEVELP]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas en el Excel: {missing}")
        st.stop()

    # Tipos
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
    df = df.dropna(subset=[COL_TIME]).sort_values(COL_TIME).reset_index(drop=True)
    df[COL_PUMPRPM] = pd.to_numeric(df[COL_PUMPRPM], errors="coerce")
    df[COL_LEVELP]  = pd.to_numeric(df[COL_LEVELP],  errors="coerce")

    return df

def remove_outliers_iqr(s: pd.Series, k: float = 3.0) -> pd.Series:
    """IQR outlier removal. Se usará solo en rpm (no en nivel para preservar rebalses)."""
    if s.dropna().empty:
        return s
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k*iqr, q3 + k*iqr
    return s.where((s >= lo) & (s <= hi))

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica. No se recorta nivel >100% (rebalse); rpm sí se limpia suavemente."""
    out = df.copy()

    # RPM: negativos -> NaN, limpieza IQR + mediana móvil
    out.loc[out[COL_PUMPRPM] < 0, COL_PUMPRPM] = np.nan
    out[COL_PUMPRPM] = remove_outliers_iqr(out[COL_PUMPRPM])
    rpm_smooth = out[COL_PUMPRPM].rolling(window=5, center=True, min_periods=1).median()
    out[COL_PUMPRPM] = rpm_smooth.interpolate(limit=5).bfill().ffill()

    # Nivel: preservamos >100% (rebalse). Negativos -> NaN
    out.loc[out[COL_LEVELP] < 0, COL_LEVELP] = np.nan

    # Index temporal (permitimos timestamps duplicados)
    out = out.set_index(COL_TIME).sort_index()
    return out

def extract_episodes(flag_series: pd.Series, min_duration_s: float = 0.0) -> pd.DataFrame:
    """
    Extrae episodios continuos (runs) a partir de una serie booleana indexada por tiempo.
    Robusto a índices con timestamps duplicados (no usa groupby con reindex).
    """
    if flag_series is None or flag_series.empty:
        return pd.DataFrame(columns=["start", "end", "duration_s", "n_samples"])

    s_bool = flag_series.fillna(False).astype(bool).to_numpy()
    if not s_bool.any():
        return pd.DataFrame(columns=["start", "end", "duration_s", "n_samples"])

    idx = pd.to_datetime(flag_series.index.to_numpy())
    change = np.cumsum(np.r_[0, s_bool[1:] != s_bool[:-1]])

    true_idx = idx[s_bool]
    true_grp = change[s_bool]

    tmp = pd.DataFrame({"t": true_idx, "g": true_grp})
    agg = tmp.groupby("g", sort=False)["t"].agg(["min", "max", "size"]).reset_index(drop=True)

    episodes = pd.DataFrame({
        "start": agg["min"],
        "end":   agg["max"],
        "duration_s": (agg["max"] - agg["min"]).dt.total_seconds(),
        "n_samples": agg["size"]
    })

    if min_duration_s > 0:
        episodes = episodes[episodes["duration_s"] >= min_duration_s].reset_index(drop=True)
    else:
        episodes = episodes.reset_index(drop=True)

    return episodes

def summarize_episodes(episodes: pd.DataFrame, label: str, total_span_s: float) -> dict:
    """
    Resumen incluyendo porcentaje del tiempo total para evaluar representatividad.
    """
    if episodes.empty or total_span_s <= 0:
        return {"label": label, "count": 0, "total_min": 0.0, "total_share_%": 0.0,
                "median_min": 0.0, "p90_min": 0.0, "max_min": 0.0}
    dmins = episodes["duration_s"] / 60.0
    total_share = 100.0 * episodes["duration_s"].sum() / total_span_s
    return {
        "label": label,
        "count": int(len(episodes)),
        "total_min": float(dmins.sum()),
        "total_share_%": float(total_share),
        "median_min": float(dmins.median()),
        "p90_min": float(dmins.quantile(0.90)),
        "max_min": float(dmins.max())
    }

# ---------- Gráficos ----------
def plot_scatter_categories(df, rpm_col, lvl_col, flags_pump, flags_over, flags_crit, thr_rpm):
    """
    Unifica categorías (Otros, Techo, Rebalse, Crítico) en un único scatter.
    """
    d = pd.DataFrame({
        "rpm": df[rpm_col],
        "lvl": df[lvl_col],
        "ceiling": flags_pump.reindex(df.index).fillna(False),
        "overflow": flags_over.reindex(df.index).fillna(False),
        "critical": flags_crit.reindex(df.index).fillna(False),
    }).dropna()

    # Categorías exclusivas
    d["cat"] = "Otros"
    d.loc[(d["ceiling"]) & (~d["overflow"]), "cat"] = "Techo"
    d.loc[(~d["ceiling"]) & (d["overflow"]), "cat"] = "Rebalse"
    d.loc[(d["ceiling"]) & (d["overflow"]), "cat"] = "Techo+Rebalse"
    d.loc[d["critical"], "cat"] = "CRÍTICO (Techo + Nivel≥99%)"

    colors = {
        "Otros": "#A0A0A0",
        "Techo": COLOR_PUMP,
        "Rebalse": COLOR_LEVEL,
        "Techo+Rebalse": "#9467bd",
        "CRÍTICO (Techo + Nivel≥99%)": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(9,6))
    for cat, sub in d.groupby("cat"):
        ax.scatter(sub["rpm"], sub["lvl"], s=10 if cat!="CRÍTICO (Techo + Nivel≥99%)" else 24,
                   alpha=0.7 if cat!="Otros" else 0.25, label=cat, c=colors.get(cat, "#999999"))
    ax.axhline(99, linestyle="--", color=COLOR_LEVEL, label="Nivel 99%")
    if np.isfinite(thr_rpm):
        ax.axvline(thr_rpm, linestyle="--", color=COLOR_PUMP, label=f"Umbral máx bomba (= {thr_rpm:.0f} rpm)")
    ax.set_title("rpm vs Nivel — categorías unificadas")
    ax.set_xlabel("Velocidad de Bomba [rpm]")
    ax.set_ylabel("Nivel del Cajón [%]")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

def plot_hexbin(df, rpm_col, lvl_col, thr_rpm):
    """
    Densidad: cada hexágono muestra cuántos puntos caen en esa celda.
    Zonas más oscuras => operación más frecuente en ese rango rpm–nivel.
    """
    df2 = df[[rpm_col, lvl_col]].dropna()
    if df2.empty:
        st.info("No hay datos suficientes para hexbin.")
        return
    fig, ax = plt.subplots(figsize=(8,5))
    hb = ax.hexbin(df2[rpm_col], df2[lvl_col], gridsize=40, mincnt=1)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Densidad (conteo de puntos)")
    ax.axhline(99, linestyle="--", color=COLOR_LEVEL, label="Nivel 99%")
    if np.isfinite(thr_rpm):
        ax.axvline(thr_rpm, linestyle="--", color=COLOR_PUMP, label=f"Umbral máx bomba (= {thr_rpm:.0f} rpm)")
    ax.set_title("Mapa de densidad rpm vs nivel (hexbin)")
    ax.set_xlabel("Velocidad de Bomba [rpm]")
    ax.set_ylabel("Nivel del Cajón [%]")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)
    st.caption("Interpretación: hexágonos más intensos indican mayor concentración de puntos (operación más frecuente).")

def plot_hist(df: pd.DataFrame, col: str, title: str, xlabel: str, color: str):
    s = df[col].dropna()
    n = int(s.shape[0])
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(s, bins=40, color=color)
    ax.set_title(f"{title} — N={n}")
    ax.set_xlabel(xlabel); ax.set_ylabel("Frecuencia")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig, clear_figure=True)

def plot_box(df: pd.DataFrame, col: str, title: str, ylabel: str, color: str):
    """
    Boxplot sin outliers y con parámetros (Q1, Mediana, Q3, bigotes) anotados dentro.
    """
    s = df[col].dropna().values
    if s.size == 0:
        st.info(f"Sin datos para {title}.")
        return

    ser = pd.Series(s)
    q1, med, q3 = ser.quantile(0.25), ser.quantile(0.50), ser.quantile(0.75)
    iqr = q3 - q1
    lower_whisker = ser[ser >= (q1 - 1.5*iqr)].min()
    upper_whisker = ser[ser <= (q3 + 1.5*iqr)].max()

    fig, ax = plt.subplots(figsize=(6,4))
    ax.boxplot(
        s, vert=True, showfliers=False,
        boxprops=dict(color=color),
        medianprops=dict(color=color, linewidth=2),
        whiskerprops=dict(color=color),
        capprops=dict(color=color)
    )
    ax.set_title(title); ax.set_ylabel(ylabel)

    # Anotaciones
    x = 1.07
    ax.text(x, q1, f"Q1={q1:.2f}", color=color, va="center")
    ax.text(x, med, f"Med={med:.2f}", color=color, va="center", fontweight="bold")
    ax.text(x, q3, f"Q3={q3:.2f}", color=color, va="center")
    ax.text(x, lower_whisker, f"W-={lower_whisker:.2f}", color=color, va="center")
    ax.text(x, upper_whisker, f"W+={upper_whisker:.2f}", color=color, va="center")

    st.pyplot(fig, clear_figure=True)

# =========================
# CARGA & ETL (antes del sidebar para poder describir rango en help)
# =========================
if not DATA_PATH.exists():
    st.error(f"No se encontró el archivo {DATA_PATH}. Súbelo como Data/dataset.xlsx.")
    st.stop()

df_raw = load_data(DATA_PATH)
df = clean_data(df_raw)

# =========================
# SIDEBAR (parámetros)
# =========================
st.sidebar.title("Parámetros")

exclude_after_cut = st.sidebar.toggle(
    "Ignorar datos post caida del Molino",
    value=False,
    help=("Excluye del análisis todo dato desde el 23-08-2025 (inclusive) en adelante. "
          "El rango resultante es desde el inicio disponible hasta 22-08-2025 23:59:59.")
)

min_event_s  = st.sidebar.slider(
    "Duración mínima de EPISODIO [s]",
    0, 600, 60, 10,
    help="Episodios continuos de menos de este tiempo se ignoran para evitar 'chispazos'."
)
high_quant   = st.sidebar.slider(
    "Cuantil para 'Nivel alto' (pXX)",
    80, 98, 90, 1,
    help="Referencia para nivel alto contextual. p90 por defecto significa que sólo el 10% de los datos supera dicho valor."
)

# Aplicar botón: excluir datos del 23-ago-2025 en adelante (inclusive)
if exclude_after_cut:
    df = df.loc[df.index < CUTOFF_DATE]

if df.empty:
    st.warning("No hay datos con la configuración actual. Revisa el botón de exclusión por parada.")
    st.stop()

# Rango final seleccionado (etiqueta visible)
range_start, range_end = df.index.min(), df.index.max()

# =========================
# 1) RANGO TEMPORAL SELECCIONADO Y ESTADÍSTICOS BÁSICOS
# =========================
st.subheader("1) Rango temporal seleccionado y Estadísticos básicos")
st.write("**Rango:**")
st.write(f"Inicio: `{range_start}`")
st.write(f"Término: `{range_end}`")

# Tabla centrada y explícita
stats = df[[COL_PUMPRPM, COL_LEVELP]].describe().loc[
    ["count","mean","std","min","25%","50%","75%","max"]
].rename(index={
    "count":"count","mean":"mean","std":"std","min":"min",
    "25%":"25%","50%":"50%","75%":"75%","max":"max"
})
st.dataframe(
    stats.style.format("{:.2f}").set_properties(**{"text-align":"center"}).set_table_styles([
        dict(selector="th", props=[("text-align","center")])
    ]),
    use_container_width=True
)

# =========================
# 2) MEMORIA DE CÁLCULO (criterios y símbolos) — LaTeX
# =========================
st.subheader("2) Memoria de cálculo (criterios y símbolos)")

st.markdown("**Criterios:**")
st.latex(r"""
\textbf{Techo de velocidad de bomba:}\quad
\mathrm{RPM}_{\text{bomba}} \;\ge\; 635\ \text{rpm}
""")
st.latex(r"""
\textbf{Rebalse del cajón:}\quad
\mathrm{Nivel} \;\ge\; 100\%
""")
st.latex(r"""
\textbf{Nivel alto (contexto):}\quad
\mathrm{Nivel} \;\ge\; P_{q}(\mathrm{Nivel}),\;\; q\in[0,1]
""")

st.markdown("**Símbolos:**")
st.latex(r"""
q:\ \text{Cuantil de referencia para definir 'nivel alto' (p.ej.\ }q=0.90\Rightarrow p90\text{)}.
""")

# =========================
# 3) MÉTRICAS DE OPERACIÓN (nueva lógica)
# =========================
st.subheader("3) Métricas de operación")

# Flags con nueva lógica
flags_pump     = (df[COL_PUMPRPM] >= PUMP_MAX_RPM)           # techo fijo = 356 rpm
flags_overflow = (df[COL_LEVELP]  >= 100.0)                   # rebalse explícito
level_pq       = df[COL_LEVELP].quantile(high_quant/100.0)    # pXX de nivel
flags_high     = (df[COL_LEVELP]  >= level_pq)
flag_lvl99     = (df[COL_LEVELP]  >= 99.0)
flag_critical  = flags_pump & flag_lvl99                       # crítico: techo + nivel ≥99%
flag_overflow_headroom = flags_overflow & (~flags_pump)        # rebalse con holgura (<356 rpm)

metrics = {
    "Umbral 'bomba al techo' [rpm]": PUMP_MAX_RPM,
    "Tiempo con 'bomba al techo' (%)": 100 * flags_pump.mean(),
    "Tiempo con rebalse (Nivel ≥ 100%) (%)": 100 * flags_overflow.mean(),
    f"Tiempo con 'nivel alto' (≥ p{high_quant}) (%)": 100 * flags_high.mean(),
    "Tiempo 'crítico' (techo + Nivel ≥ 99%) (%)": 100 * flag_critical.mean(),
    "Tiempo rebalse con holgura (<356 rpm) (%)": 100 * flag_overflow_headroom.mean(),
}
def g(x: str) -> str:
    return f"<span style='color:#10B981;font-weight:600'>{x}</span>"

time_ceiling   = metrics["Tiempo con 'bomba al techo' (%)"]
time_overflow  = metrics["Tiempo con rebalse (Nivel ≥ 100%) (%)"]
time_high      = metrics[f"Tiempo con 'nivel alto' (≥ p{high_quant}) (%)"]
time_critical  = metrics["Tiempo 'crítico' (techo + Nivel ≥ 99%) (%)"]
time_over_hdg  = metrics["Tiempo rebalse con holgura (<356 rpm) (%)"]

phrase = (
    f"Con techo fijo {g('356 rpm')}, la bomba estuvo 'al techo' {g(f'{time_ceiling:.1f}%')}. "
    f"El cajón rebasó (≥100%) {g(f'{time_overflow:.1f}%')}, y superó p{high_quant} {g(f'{time_high:.1f}%')}. "
    f"Las condiciones {g('críticas')} (techo + Nivel≥99%) ocurrieron {g(f'{time_critical:.1f}%')}. "
    f"Hubo {g(f'{time_over_hdg:.1f}%')} del tiempo con rebalse mientras la bomba aún tenía holgura (<635 rpm)."
)
st.markdown(phrase, unsafe_allow_html=True)

# =========================
# 4) EPISODIOS (duración continua) — explicación + porcentajes
# =========================
st.subheader("4) Episodios (duración continua)")
st.markdown("""
**Definiciones e implicaciones**  
- **Bomba al techo**: períodos continuos con rpm ≥ 356. Implica posible restricción de capacidad de evacuación por velocidad.  
- **Rebalse (Nivel ≥ 100%)**: períodos continuos con nivel en o sobre el borde del cajón. Implica riesgo operativo y potencial derrame.  
- **Techo + Rebalse**: co-ocurrencia de velocidad al techo y rebalse; sugiere límite de capacidad de evacuación en condiciones exigentes.  
- **Crítico (Techo + Nivel ≥ 99%)**: similar al anterior, ampliando sensibilidad a nivel muy alto (≥99%).  
- **Rebalse con holgura**: rebalse mientras la bomba **no** está al techo (<356 rpm); sugiere oportunidades de control/operación (p. ej., aumentar setpoint o revisar lógica).
""")

total_span_s = (range_end - range_start).total_seconds()
ep_pump      = extract_episodes(flags_pump,                    min_duration_s=min_event_s)
ep_over      = extract_episodes(flags_overflow,                min_duration_s=min_event_s)
ep_both      = extract_episodes(flags_pump & flags_overflow,   min_duration_s=min_event_s)
ep_critical  = extract_episodes(flag_critical,                 min_duration_s=min_event_s)
ep_over_hdg  = extract_episodes(flag_overflow_headroom,        min_duration_s=min_event_s)

summary = pd.DataFrame([
    summarize_episodes(ep_pump,     "Bomba al techo",                 total_span_s),
    summarize_episodes(ep_over,     "Rebalse (Nivel ≥ 100%)",         total_span_s),
    summarize_episodes(ep_both,     "Bomba al techo + Rebalse",       total_span_s),
    summarize_episodes(ep_critical, "CRÍTICO: Techo + Nivel ≥ 99%",   total_span_s),
    summarize_episodes(ep_over_hdg, "Rebalse con holgura (<356 rpm)", total_span_s),
])
st.write(summary)

# Minutos por día (interactivo, sin título dentro del gráfico)
st.markdown("**Minutos por día (interactivo): Crítico y Rebalse**")
try:
    # Resample a 1min
    crit_1min = flag_critical.resample("1min").max()
    over_1min = flags_overflow.resample("1min").max()

    # Agregar por día
    df_day = pd.DataFrame({
        "date": pd.to_datetime(crit_1min.index.date),
        "crit_min": crit_1min.values.astype(int),
        "over_min": over_1min.values.astype(int),
    }).groupby("date", as_index=False).sum()

    fig_bar = px.bar(df_day, x="date", y=["crit_min", "over_min"],
                     labels={"value":"Minutos", "date":"Día", "variable":""})
    fig_bar.update_layout(showlegend=True, title_text=None)  # sin título
    st.plotly_chart(fig_bar, use_container_width=True)
except Exception as e:
    st.info(f"No fue posible calcular minutos por día: {e}")

# =========================
# 5) Gráficos (títulos + explicaciones)
# =========================
st.subheader("5) Gráficos")

st.markdown("### Dispersión por categorías")
st.caption("Muestra la relación rpm–nivel clasificando puntos en: Techo, Rebalse, Techo+Rebalse y Crítico.")
plot_scatter_categories(df, COL_PUMPRPM, COL_LEVELP, flags_pump, flags_overflow, flag_critical, PUMP_MAX_RPM)

st.markdown("### Mapa de densidad rpm–nivel (hexbin)")
st.caption("Indica las zonas de operación más frecuentes; útil para identificar regímenes habituales y extremos.")
plot_hexbin(df, COL_PUMPRPM, COL_LEVELP, PUMP_MAX_RPM)

c3, c4 = st.columns(2)
with c3:
    st.markdown("### Box & Whisker — Nivel del Cajón")
    st.caption("Sin outliers; se anotan Q1, Mediana, Q3 y bigotes (Tukey).")
    plot_box(df, COL_LEVELP, "Nivel del Cajón [%]", "%", COLOR_LEVEL)

with c4:
    st.markdown("### Box & Whisker — Velocidad de Bomba")
    st.caption("Sin outliers; se anotan Q1, Mediana, Q3 y bigotes (Tukey).")
    plot_box(df, COL_PUMPRPM, "Velocidad de Bomba [rpm]", "rpm", COLOR_PUMP)

c5, c6 = st.columns(2)
with c5:
    st.markdown("### Histograma — Velocidad de Bomba")
    st.caption("Distribución de rpm con N total de muestras.")
    plot_hist(df, COL_PUMPRPM, "Velocidad de Bomba [rpm]", "rpm", COLOR_PUMP)
with c6:
    st.markdown("### Histograma — Nivel del Cajón")
    st.caption("Distribución de nivel [%] con N total de muestras.")
    plot_hist(df, COL_LEVELP, "Nivel del Cajón [%]", "%", COLOR_LEVEL)

# =========================
# 6) Zoom a episodio (selector intuitivo)
# =========================
st.subheader("6) Zoom a episodio (máx + ≥99%, Rebalse o Techo)")
st.caption(
    "Selecciona el **tipo de episodio** y luego el episodio específico. "
    "La ventana permite ver el contexto antes y después."
)

# Preparar catálogos legibles de episodios
def episode_labels(df_ep, label):
    if df_ep.empty:
        return []
    rows = []
    for i, r in df_ep.iterrows():
        dur = r["duration_s"]/60.0
        rows.append((i, f"{label} • {r['start']} → {r['end']} • {dur:.1f} min"))
    return rows

ep_type = st.radio("Tipo de episodio", ["Crítico (Techo + ≥99%)", "Rebalse (≥100%)", "Techo (356 rpm)"])
if ep_type == "Crítico (Techo + ≥99%)":
    ep_pool = ep_critical
    label = "Crítico"
elif ep_type == "Rebalse (≥100%)":
    ep_pool = ep_over
    label = "Rebalse"
else:
    ep_pool = ep_pump
    label = "Techo"

labels = episode_labels(ep_pool, label)
if labels:
    idx_default = 0
    idx_map = {txt:i for i, txt in labels}
    choice = st.selectbox("Selecciona episodio", options=[txt for _, txt in labels], index=idx_default)
    idx_sel = idx_map[choice]
    win_min = st.slider("Ventana alrededor del episodio [min]", 1, 60, 15, 1)

    estart = ep_pool.loc[idx_sel, "start"]
    eend   = ep_pool.loc[idx_sel, "end"]
    wstart = estart - pd.Timedelta(minutes=win_min)
    wend   = eend   + pd.Timedelta(minutes=win_min)
    seg = df.loc[wstart:wend]

    fig, ax1 = plt.subplots(figsize=(12,4))

    # --- curva de bomba (azul) y techo ---
    ax1.plot(seg.index, seg[COL_PUMPRPM], label="RPM bomba", color=COLOR_PUMP)
    ax1.axhline(PUMP_MAX_RPM, linestyle="--", color=COLOR_PUMP, label=f"Techo (= {PUMP_MAX_RPM:.0f} rpm)")
    ax1.set_ylabel("rpm", color=COLOR_PUMP)
    ax1.tick_params(axis='y', labelcolor=COLOR_PUMP)
    ax1.grid(alpha=0.3)

    # --- nivel (rojo), 99% y 100% ---
    ax2 = ax1.twinx()
    ax2.plot(seg.index, seg[COL_LEVELP], alpha=0.7, label="Nivel [%]", linestyle=":", color=COLOR_LEVEL)
    ax2.axhline(99,  linestyle="--", color=COLOR_LEVEL, label="Nivel 99%")
    ax2.axhline(100, linestyle="-.", color=COLOR_LEVEL, label="Nivel 100%")
    ax2.set_ylabel("%", color=COLOR_LEVEL)
    ax2.tick_params(axis='y', labelcolor=COLOR_LEVEL)

    # --- título del gráfico ---
    ax1.set_title(f"{label}: {estart} → {eend}  ({(eend-estart).total_seconds()/60:.1f} min)")

    # --- LEYENDA FUERA DEL GRÁFICO (a la derecha) ---
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles = h1 + h2
    labels_ = l1 + l2

    # deja espacio a la derecha y ubica la leyenda fuera
    fig.subplots_adjust(right=0.78)
    fig.legend(handles, labels_, loc="center left", bbox_to_anchor=(0.80, 0.5),
            frameon=True, title="Leyenda")

    st.pyplot(fig, clear_figure=True)

else:
    st.info("No hay episodios para el tipo seleccionado.")

# =========================
# 7) Conclusiones
# =========================
st.subheader("7) Conclusiones")
concl = []

if time_ceiling > 15:
    concl.append("• La bomba presenta **limitación por velocidad** relevante (>15% del tiempo).")
else:
    concl.append("• La bomba **no** muestra limitación por velocidad significativa en términos agregados.")

if time_overflow > 5:
    concl.append("• El cajón **rebalsa con frecuencia** (Nivel ≥ 100% > 5% del tiempo).")
elif time_overflow > 0:
    concl.append("• Se registran **eventos de rebalse** puntuales.")

if time_critical > 2 or len(ep_critical) > 0:
    concl.append("• Existen **episodios críticos** (techo + ≥99%) que sugieren revisar control/estrategia de evacuación.")
else:
    concl.append("• No se observan co-ocurrencias críticas de **techo + ≥99%**.")

if time_over_hdg > 0.5 or len(ep_over_hdg) > 0:
    concl.append("• Se detectó **rebalse con holgura** (<356 rpm); existe espacio para **aumentar caudal** (ajuste de setpoints/estrategia).")

st.write("\n".join(concl))

# =========================
# 8) Reporte (único)
# =========================
st.subheader("8) Reporte")
reporte = []
reporte.append("# Reporte PU-010 — Análisis Bomba vs Nivel")
reporte.append("")
reporte.append(f"**Rango analizado:** {range_start} → {range_end}")
reporte.append(f"**Exclusión post caída de Molino:** {'Sí' if exclude_after_cut else 'No'} (corte en 23-08-2025)")
reporte.append("")
reporte.append("## Métricas")
reporte.append(f"- Umbral 'bomba al techo': {PUMP_MAX_RPM:.0f} rpm")
reporte.append(f"- Tiempo 'bomba al techo': {time_ceiling:.1f}%")
reporte.append(f"- Tiempo rebalse (Nivel ≥100%): {time_overflow:.1f}%")
reporte.append(f"- Tiempo nivel alto (≥ p{high_quant}): {time_high:.1f}%")
reporte.append(f"- Tiempo CRÍTICO (techo + Nivel ≥99%): {time_critical:.1f}%")
reporte.append(f"- Tiempo rebalse con holgura (<356 rpm): {time_over_hdg:.1f}%")
reporte.append("")
reporte.append("## Episodios (duración continua)")
def fmt_row(d):
    return (f"- {d['label']}: {d['count']} episodios, total {d['total_min']:.1f} min "
            f"({d['total_share_%']:.2f}% del período), mediana {d['median_min']:.1f} min, "
            f"p90 {d['p90_min']:.1f} min, máx {d['max_min']:.1f} min")
for _, r in summary.iterrows():
    reporte.append(fmt_row(r))
reporte.append("")
reporte.append("## Conclusiones")
for c in concl:
    reporte.append(f"- {c.replace('• ', '')}")

report_text = "\n".join(reporte)
st.download_button(
    "Descargar reporte (Markdown)",
    report_text,
    file_name="reporte_PU010.md"
)

st.caption("Colores consistentes: azul = bomba (rpm), rojo = cajón (nivel). Los rebalses (Nivel ≥ 100%) se contabilizan explícitamente sin recorte.")
