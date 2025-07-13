import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import re  # Add re import for regex
import os
import joblib
import json
from datetime import datetime, timedelta
import numpy as np
from catboost import Pool
import io
import subprocess
import sys

# Install spacy model if not available
try:
    import spacy
    nlp = spacy.load("es_core_news_sm")
except OSError:
    st.info("Installing Spanish language model for spacy...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])
    import spacy
    nlp = spacy.load("es_core_news_sm")


# Configuración estilo gráfico general (sin líneas de fondo)
plt.rcParams.update({
    "axes.edgecolor": "#E0E0E0",
    "axes.linewidth": 0.8,
    "axes.titlesize": 14,
    "axes.titleweight": 'bold',
    "axes.labelcolor": "#333333",
    "axes.labelsize": 12,
    "xtick.color": "#666666",
    "ytick.color": "#666666",
    "font.family": "DejaVu Sans",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.autolayout": True,
    "figure.constrained_layout.use": True
})

# Paletas de colores personalizadas
COLOR_GRADIENT = ["#e6f3ff", "#cce7ff", "#99cfff", "#66b8ff", "#33a0ff", "#0088ff", "#006acc", "#004d99", "#003366"]
TEMPORADA_COLORS = ["#e6f3ff", "#99ccff", "#4d94ff", "#0066cc", "#004d99", "#003366", "#001a33", "#000d1a", "#000000"]
COLOR_GRADIENT_WARM = ["#fff5e6", "#ffebcc", "#ffd699", "#ffc266", "#ffad33", "#ff9900", "#cc7a00", "#995c00", "#663d00"]
COLOR_GRADIENT_GREEN = ["#e6ffe6", "#ccffcc", "#99ff99", "#66ff66", "#33ff33", "#00ff00", "#00cc00", "#009900", "#006600"]

TIENDAS_EXTRANJERAS = [
    "I301COINBERGAMO(TRUCCO)", "I302COINVARESE(TRUCCO)", "I303COINBARICASAMASSIMA(TRUCCO)",
    "I304COINMILANO5GIORNATE(TRUCCO)", "I305COINROMACINECITTA(TRUCCO)", "I306COINGENOVA(TRUCCO)",
    "I309COINSASSARI(TRUCCO)", "I314COINCATANIA(TRUCCO)", "I315COINCAGLIARI(TRUCCO)",
    "I316COINLECCE(TRUCCO)", "I317COINMILANOCANTORE(TRUCCO)", "I318COINMESTRE(TRUCCO)",
    "I319COINPADOVA(TRUCCO)", "I320COINFIRENZE(TRUCCO)", "I321COINROMASANGIOVANNI(TRUCCO)",
    "TRUCCOONLINEB2C"
]

COL_ONLINE = '#2ca02c'   # verde fuerte
COL_OTRAS = '#ff7f0e'    # naranja

def custom_sort_key(talla):
    """
    Clave de ordenación personalizada para tallas.
    Prioriza: 1. Tallas numéricas, 2. Tallas de letra estándar, 3. Tallas únicas, 4. Resto.
    """
    talla_str = str(talla).upper().strip()
    
    # Prioridad 1: Tallas numéricas (e.g., '36', '38')
    if talla_str.isdigit():
        return (0, int(talla_str))
    
    # Prioridad 2: Tallas de letra estándar
    size_order = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
    if talla_str in size_order:
        return (1, size_order.index(talla_str))
        
    # Prioridad 3: Tallas únicas
    if talla_str in ['U', 'ÚNICA', 'UNICA', 'TU']:
        return (2, talla_str)
        
    # Prioridad 4: Resto, ordenado alfabéticamente
    return (3, talla_str)

def setup_streamlit_styles():
    """Configurar estilos de Streamlit"""
    st.markdown("""
    <style>
    .dashboard-container {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: white;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    .kpi-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
        width: 100%;
        margin-top: 0;
        padding-top: 0;
    }
    .kpi-row {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        flex-wrap: nowrap;
        width: 100%;
    }
    .kpi-group {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        margin-top: 0;
        background-color: white;
        width: 100%;
    }
    .kpi-group-title {
        color: #666666;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 10px;
        margin-top: 0;
        padding-bottom: 5px;
        border-bottom: 1px solid #e5e7eb;
    }
    .kpi-item {
        flex: 1;
        text-align: center;
        padding: 15px;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        background-color: white;
        min-width: 150px;
    }
    .small-font {
        color: #666666;
        font-size: 14px;
        margin-bottom: 5px;
        margin-top: 0;
    }
    .metric-value {
        color: #111827;
        font-size: 24px;
        font-weight: bold;
        margin: 0;
    }
    .section-title {
        color: #111827;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 20px;
        margin-top: 0;
        line-height: 1.2;
    }
    .viz-title {
        color: #111827;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 15px;
        margin-top: 0;
        line-height: 1.2;
    }
    .viz-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 20px;
        margin-top: 20px;
    }
    .viz-container {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 20px;
        background-color: white;
        height: 100%;
    }
    div.block-container {
        padding-top: 0;
        margin-top: 0;
    }
    div.stMarkdown {
        margin-top: 0;
        padding-top: 0;
    }
    </style>
    """, unsafe_allow_html=True)

def viz_title(text):
    """Función unificada para títulos de visualizaciones"""
    st.markdown(f'<h3 class="viz-title">{text}</h3>', unsafe_allow_html=True)

def titulo(text):
    st.markdown(f"<h4 style='text-align:left;color:#666666;margin:0;padding:0;font-size:20px;font-weight:bold;'>{text}</h4>", unsafe_allow_html=True)

def subtitulo(text):
    st.markdown(f"<h5 style='text-align:left;color:#666666;margin:0;padding:0;font-size:22px;font-weight:bold;'>{text}</h5>", unsafe_allow_html=True)

def aplicar_filtros(df_ventas, df_traspasos):
    if not pd.api.types.is_datetime64_any_dtype(df_ventas['Fecha venta']):
        df_ventas['Fecha venta'] = pd.to_datetime(df_ventas['Fecha venta'], format='%d/%m/%Y', errors='coerce')
    fecha_min, fecha_max = df_ventas['Fecha venta'].min(), df_ventas['Fecha venta'].max()

    fecha_inicio, fecha_fin = st.sidebar.date_input(
        "Rango de fechas",
        [fecha_min, fecha_max],
        min_value=fecha_min,
        max_value=fecha_max
    )

    if fecha_inicio > fecha_fin:
        st.sidebar.error("La fecha de inicio debe ser anterior a la fecha de fin.")
        if df_traspasos is not None:
            return df_ventas.iloc[0:0], df_traspasos.iloc[0:0], False, []
        return df_ventas.iloc[0:0], False, []

    df_ventas_filtrado = df_ventas[(df_ventas['Fecha venta'] >= pd.to_datetime(fecha_inicio)) &
                     (df_ventas['Fecha venta'] <= pd.to_datetime(fecha_fin))]
    tiendas = sorted(df_ventas_filtrado['Tienda'].dropna().unique())
    modo_tienda = st.sidebar.selectbox(
        "Modo selección tiendas",
        ["Todas las tiendas", "Seleccionar tiendas específicas"]
    )
    if modo_tienda == "Todas las tiendas":
        tienda_seleccionada = tiendas
        tiendas_especificas = False
    else:
        tienda_seleccionada = st.sidebar.multiselect(
            "Selecciona tienda(s)",
            options=tiendas
        )
        if not tienda_seleccionada:
            st.sidebar.warning("Selecciona al menos una tienda para mostrar datos.")
            if df_traspasos is not None:
                return df_ventas.iloc[0:0], df_traspasos.iloc[0:0], False, []
            return df_ventas.iloc[0:0], False, []
        tiendas_especificas = True
    
    df_ventas_filtrado = df_ventas_filtrado[df_ventas_filtrado['Tienda'].isin(tienda_seleccionada)]
    
    # Aplicar filtro de tienda a traspasos si se proporciona
    if df_traspasos is not None:
        df_traspasos_filtrado = df_traspasos.copy()
        df_traspasos_filtrado['Fecha enviado'] = pd.to_datetime(df_traspasos_filtrado['Fecha enviado'], format='%d/%m/%Y', errors='coerce')
        # Asegurar que la columna Tienda existe en traspasos
        if 'Tienda' in df_traspasos_filtrado.columns:
            df_traspasos_filtrado = df_traspasos_filtrado[df_traspasos_filtrado['Tienda'].isin(tienda_seleccionada)]
        return df_ventas_filtrado, df_traspasos_filtrado, tiendas_especificas, tienda_seleccionada
    
    return df_ventas_filtrado, tiendas_especificas, tienda_seleccionada



def create_resizable_chart(chart_key, chart_function):
    """
    Crea un contenedor para el gráfico con funcionalidad de redimensionamiento
    """
    col1, col2 = st.columns([4, 1])
    with col1:
        size = st.select_slider(
            f'Ajustar tamaño del gráfico {chart_key}',
            options=['Pequeño', 'Mediano', 'Grande', 'Extra Grande'],
            value='Mediano',
            key=f'size_{chart_key}'
        )
    
    sizes = {
        'Pequeño': 300,
        'Mediano': 500,
        'Grande': 700,
        'Extra Grande': 900
    }
    
    height = sizes[size]
    
    st.markdown(f'<div class="chart-container" style="height: {height}px;">', unsafe_allow_html=True)
    chart_function(height)
    st.markdown('</div>', unsafe_allow_html=True)
def plot_bar(df, x, y, title, palette='Greens', rotate_x=30, color=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    if color:
        sns.barplot(x=x, y=y, data=df, color=color, ax=ax)
    else:
        # Normalizar los valores para el degradado
        norm_values = (df[y] - df[y].min()) / (df[y].max() - df[y].min())
        colors = [COLOR_GRADIENT[int(v * (len(COLOR_GRADIENT)-1))] if not pd.isna(v) else COLOR_GRADIENT[0] for v in norm_values]
        
        sns.barplot(x=x, y=y, data=df, palette=colors, ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold', color="#111827", loc="left", pad=0)
    ax.set_xlabel(x, fontsize=13)
    ax.set_ylabel(y, fontsize=13)
    plt.xticks(rotation=rotate_x, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    ax.grid(False)
    ax.set_axisbelow(True)
    sns.despine()
    
    # Ajustar valores sobre las barras
    for bar in ax.patches:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.01 * value),
            f'{int(value)}',
            ha='center',
            va='bottom',
            fontsize=10,
            color='#333'
        )
    
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)

def viz_container(title, render_function):
    """Contenedor para visualizaciones"""
    st.markdown('<div class="viz-container">', unsafe_allow_html=True)
    viz_title(title)
    render_function()
    st.markdown('</div>', unsafe_allow_html=True)

def mostrar_dashboard(df_productos, df_traspasos, df_ventas, seccion):
    setup_streamlit_styles()
    
    # Use cached preprocessing for better performance
    df_ventas = preprocess_ventas_data(df_ventas)
    df_productos = preprocess_productos_data(df_productos)
    df_traspasos = preprocess_traspasos_data(df_traspasos)
    
    # Merge Precio Coste from df_productos into df_ventas using Código único
    df_ventas = df_ventas.merge(
        df_productos[['Código único', 'Precio Coste']],
        on='Código único',
        how='left',
        suffixes=('', '_producto')
    )
    # Prefer Precio Coste from df_productos if available
    if 'Precio Coste_producto' in df_ventas.columns:
        df_ventas['Precio Coste'] = df_ventas['Precio Coste_producto'].combine_first(df_ventas['Precio Coste'])
        df_ventas = df_ventas.drop(columns=['Precio Coste_producto'])

   
    # Calcular ranking completo de todas las tiendas ANTES de aplicar filtros
    ventas_por_tienda_completo = calculate_store_rankings(df_ventas)
    
    # Aplicar filtros
    df_ventas, df_traspasos_filtrado, tiendas_especificas, tienda_seleccionada = aplicar_filtros(df_ventas, df_traspasos)
    if df_ventas.empty:
        st.warning("No hay datos para mostrar con los filtros seleccionados.")
        return

    if seccion == "Resumen General":
        try:
            # Calcular KPIs
            total_ventas_dinero = df_ventas['Beneficio'].sum()
            total_familias = df_ventas['Familia'].nunique()
            
            # Calcular Total Devoluciones (monetary amount of negative quantities)
            devoluciones = df_ventas[df_ventas['Cantidad'] < 0].copy()
            total_devoluciones_dinero = abs(devoluciones['Beneficio'].sum())  # Use abs() to show positive value
            
            # Separar tiendas físicas y online
            ventas_fisicas = df_ventas[~df_ventas['Es_Online']]
            ventas_online = df_ventas[df_ventas['Es_Online']]
            
            # Calcular KPIs por tipo de tienda
            ventas_fisicas_dinero = ventas_fisicas['Beneficio'].sum()
            ventas_online_dinero = ventas_online['Beneficio'].sum()
            tiendas_fisicas = ventas_fisicas['Tienda'].nunique()
            tiendas_online = ventas_online['Tienda'].nunique()

            # KPIs Generales en una sola fila
            st.markdown("""
                <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: white;">
                    <div style="color: #666666; font-size: 16px; font-weight: 600; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #e5e7eb;">
                        KPIs Generales
                    </div>
                    <div style="display: flex; justify-content: space-between; gap: 15px;">
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Total Ventas Netas</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{:,.2f}€</p>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Total Devoluciones</p>
                            <p style="color: #dc2626; font-size: 24px; font-weight: bold; margin: 0;">{:,.2f}€</p>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Número de Familias</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{}</p>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Total Tiendas</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{}</p>
                        </div>
                    </div>
                </div>
            """.format(total_ventas_dinero, total_devoluciones_dinero, total_familias, tiendas_fisicas + tiendas_online), unsafe_allow_html=True)
            
            # KPIs por Tipo de Tienda en una sola fila
            st.markdown("""
                <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: white;">
                    <div style="color: #666666; font-size: 16px; font-weight: 600; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #e5e7eb;">
                        KPIs por Tipo de Tienda
                    </div>
                    <div style="display: flex; justify-content: space-between; gap: 15px;">
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Tiendas Físicas</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{}</p>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Ventas Físicas</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{:,.2f}€</p>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Tiendas Online</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{}</p>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Ventas Online</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{:,.2f}€</p>
                        </div>
                    </div>
                </div>
            """.format(tiendas_fisicas, ventas_fisicas_dinero, tiendas_online, ventas_online_dinero), unsafe_allow_html=True)
            
           
            
            # --- Rotación de stock (OPTIMIZADA) ---
            (
                tienda_mayor_rotacion, tienda_mayor_rotacion_dias, tienda_menor_rotacion, tienda_menor_rotacion_dias,
                producto_mayor_rotacion, producto_mayor_rotacion_dias, producto_menor_rotacion, producto_menor_rotacion_dias,
                promedio_global, mediana_global, std_global, total_productos_rotacion
            ) = calculate_rotation_metrics(df_productos, df_traspasos, df_ventas)

            if tienda_mayor_rotacion is not None:
                # Mostrar KPIs de rotación optimizados
                st.markdown("""
                    <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: white;">
                        <div style="color: #666666; font-size: 16px; font-weight: 600; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #e5e7eb;">
                            KPIs de Rotación de Stock (Optimizados)
                        </div>
                        <div style="display: flex; justify-content: space-between; gap: 15px; flex-wrap: wrap;">
                            <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white; min-width: 200px;">
                                <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Tienda Mayor Rotación</p>
                                <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{}</p>
                                <p style="color: #059669; font-size: 12px; margin: 0;">{:.1f} días mediana</p>
                            </div>
                            <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white; min-width: 200px;">
                                <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Tienda Menor Rotación</p>
                                <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{}</p>
                                <p style="color: #dc2626; font-size: 12px; margin: 0;">{:.1f} días mediana</p>
                            </div>
                            <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white; min-width: 200px;">
                                <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Producto Mayor Rotación</p>
                                <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{}</p>
                                <p style="color: #059669; font-size: 12px; margin: 0;">{:.1f} días mediana</p>
                            </div>
                            <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white; min-width: 200px;">
                                <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Producto Menor Rotación</p>
                                <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{}</p>
                                <p style="color: #dc2626; font-size: 12px; margin: 0;">{:.1f} días mediana</p>
                            </div>
                        </div>
                        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e5e7eb;">
                            <div style="display: flex; justify-content: space-between; gap: 15px; flex-wrap: wrap;">
                                <div style="flex: 1; text-align: center; padding: 10px; border: 1px solid #e5e7eb; border-radius: 6px; background-color: #f9fafb; min-width: 150px;">
                                    <p style="color: #666666; font-size: 12px; margin: 0 0 3px 0;">Promedio Global</p>
                                    <p style="color: #111827; font-size: 16px; font-weight: bold; margin: 0;">{:.1f} días</p>
                                </div>
                                <div style="flex: 1; text-align: center; padding: 10px; border: 1px solid #e5e7eb; border-radius: 6px; background-color: #f9fafb; min-width: 150px;">
                                    <p style="color: #666666; font-size: 12px; margin: 0 0 3px 0;">Mediana Global</p>
                                    <p style="color: #111827; font-size: 16px; font-weight: bold; margin: 0;">{:.1f} días</p>
                                </div>
                                <div style="flex: 1; text-align: center; padding: 10px; border: 1px solid #e5e7eb; border-radius: 6px; background-color: #f9fafb; min-width: 150px;">
                                    <p style="color: #666666; font-size: 12px; margin: 0 0 3px 0;">Desv. Estándar</p>
                                    <p style="color: #111827; font-size: 16px; font-weight: bold; margin: 0;">{:.1f} días</p>
                                </div>
                                <div style="flex: 1; text-align: center; padding: 10px; border: 1px solid #e5e7eb; border-radius: 6px; background-color: #f9fafb; min-width: 150px;">
                                    <p style="color: #666666; font-size: 12px; margin: 0 0 3px 0;">Total Productos</p>
                                    <p style="color: #111827; font-size: 16px; font-weight: bold; margin: 0;">{:,}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                """.format(
                    tienda_mayor_rotacion, tienda_mayor_rotacion_dias,
                    tienda_menor_rotacion, tienda_menor_rotacion_dias,
                    producto_mayor_rotacion, producto_mayor_rotacion_dias,
                    producto_menor_rotacion, producto_menor_rotacion_dias,
                    promedio_global, mediana_global, std_global, total_productos_rotacion
                ), unsafe_allow_html=True)
                
                # Mostrar estadísticas adicionales optimizadas
                st.info(f"📊 Análisis basado en {total_productos_rotacion:,} productos con rotación calculada (filtrado de outliers: 0-365 días)")
            else:
                st.info("No hay datos de entrada en almacén disponibles para calcular rotación de stock.")

            # Col 1: Ventas por mes (centered)
            col1a, col1b, col1c = st.columns([1, 2, 1])
            
            with col1b:
                viz_title("Ventas Mensuales por Tipo de Tienda")
                ventas_mes_tipo = df_ventas.groupby(['Mes', 'Es_Online']).agg({
                    'Cantidad': 'sum',
                    'Beneficio': 'sum'
                }).reset_index()
                
                ventas_mes_tipo['Tipo'] = ventas_mes_tipo['Es_Online'].map({True: 'Online', False: 'Física'})
                
                # Calculate dynamic width based on number of months
                num_months = len(ventas_mes_tipo['Mes'].unique())
                dynamic_width = max(800, num_months * 300)  # Minimum 800px, 300px per month for much wider graph
                
                fig = px.bar(ventas_mes_tipo, 
                            x='Mes', 
                            y='Cantidad', 
                            color='Tipo',
                            color_discrete_map={'Física': '#1e3a8a', 'Online': '#60a5fa'},
                            barmode='stack',
                            text='Cantidad',
                            height=400,
                            width=dynamic_width)
                
                fig.update_layout(
                    xaxis_title="Mes",
                    yaxis_title="Cantidad",
                    showlegend=True,
                    xaxis_tickangle=45,
                    margin=dict(t=0, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                
                fig.update_traces(
                    texttemplate='%{text:,.0f}', 
                    textposition='outside',
                    hovertemplate="Mes: %{x}<br>Cantidad: %{text:,.0f}<br>Ventas: %{customdata:,.2f}€<extra></extra>",
                    customdata=ventas_mes_tipo['Beneficio'],
                    opacity=0.8
                )
                
                # Use HTML container with dynamic width
                st.markdown(f"""
                    <div style="width: {dynamic_width}px; max-width: 100%; overflow-x: auto;">
                        <div style="width: 100%;">
                """, unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=False)
                st.markdown("</div></div>", unsafe_allow_html=True)

            # Col 2,3: Ranking de tiendas
            col2, col3 = st.columns(2)
            
            if tiendas_especificas:
                # Mostrar tabla de ranking para las tiendas seleccionadas (full width)
                viz_title("Ranking de Tiendas Seleccionadas")
                
                # Filtrar solo las tiendas seleccionadas del ranking completo
                tiendas_ranking = ventas_por_tienda_completo[ventas_por_tienda_completo['Tienda'].isin(tienda_seleccionada)].copy()
                
                # Calcular la familia más vendida para cada tienda (cached)
                familias_por_tienda = calculate_family_rankings(df_ventas)
                
                # Obtener la familia top para cada tienda seleccionada
                familias_top = []
                for tienda in tiendas_ranking['Tienda']:
                    familias_tienda = familias_por_tienda[familias_por_tienda['Tienda'] == tienda]
                    if not familias_tienda.empty:
                        familia_top = familias_tienda.iloc[0]['Familia']
                        familias_top.append(familia_top)
                    else:
                        familias_top.append('Sin datos')
                
                tiendas_ranking['Familia Top'] = familias_top
                
                # Reordenar columnas
                tiendas_ranking = tiendas_ranking[['Tienda', 'Ranking', 'Unidades Vendidas', 'Beneficio', 'Familia Top']]
                
                # Mostrar tabla
                st.dataframe(
                    tiendas_ranking.style.format({
                        'Unidades Vendidas': '{:,.0f}',
                        'Beneficio': '{:,.2f}€'
                    }),
                    use_container_width=True
                )
            else:
                # Mostrar top 20 y bottom 20 como antes
                with col2:
                    # Top 20 tiendas con más ventas por Beneficio
                    viz_title("Top 20 tiendas con más ventas")
                    top_20_tiendas = ventas_por_tienda_completo.head(20)
                    
                    fig = px.bar(
                        top_20_tiendas,
                        x='Tienda',
                        y='Beneficio',
                        color='Beneficio',
                        color_continuous_scale=COLOR_GRADIENT,
                        height=400,
                        labels={'Tienda': 'Tienda', 'Beneficio': 'Beneficio', 'Unidades Vendidas': 'Unidades'}
                    )
                    fig.update_layout(
                        xaxis_tickangle=45,
                        showlegend=False,
                        margin=dict(t=0, b=0, l=0, r=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    fig.update_traces(
                        texttemplate='%{y:,.2f}€',
                        textposition='outside',
                        hovertemplate="Tienda: %{x}<br>Ventas: %{y:,.2f}€<br>Unidades: %{customdata:,}<extra></extra>",
                        customdata=top_20_tiendas['Unidades Vendidas'],
                        opacity=0.8
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col3:
                    # Top 20 tiendas con menos ventas por Beneficio
                    viz_title("Top 20 tiendas con menos ventas")
                    bottom_20_tiendas = ventas_por_tienda_completo.tail(20)
                    
                    fig = px.bar(
                        bottom_20_tiendas,
                        x='Tienda',
                        y='Beneficio',
                        color='Beneficio',
                        color_continuous_scale=COLOR_GRADIENT,
                        height=400,
                        labels={'Tienda': 'Tienda', 'Beneficio': 'Beneficio', 'Unidades Vendidas': 'Unidades'}
                    )
                    fig.update_layout(
                        xaxis_tickangle=45,
                        showlegend=False,
                        margin=dict(t=0, b=0, l=0, r=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    fig.update_traces(
                        texttemplate='%{y:,.2f}€',
                        textposition='outside',
                        hovertemplate="Tienda: %{x}<br>Ventas: %{y:,.2f}€<br>Unidades: %{customdata:,}<extra></extra>",
                        customdata=bottom_20_tiendas['Unidades Vendidas'],
                        opacity=0.8
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Col 4: Unidades Vendidas por Talla (centered)
            col4a, col4b, col4c = st.columns([1, 2, 1])
            
            with col4b:
                viz_title("Unidades Vendidas por Talla")
                
                # Usar los datos ya filtrados por familia desde el sidebar
                if df_ventas.empty:
                    st.warning("No hay datos de ventas para la familia seleccionada.")
                else:
                    # Normalizar las tallas para asegurar consistencia
                    df_ventas_temp = df_ventas.copy()
                    df_ventas_temp['Talla'] = df_ventas_temp['Talla'].astype(str).str.upper().str.strip()
                    
                    #  Verificar tallas por familia
                    if not df_ventas_temp.empty:
                        familia_actual = df_ventas_temp['Familia'].iloc[0] if 'Familia' in df_ventas_temp.columns else 'Sin Familia'
                        tallas_por_familia = df_ventas_temp[df_ventas_temp['Familia'] == familia_actual]['Talla'].unique()
                        
                    
                    # Agrupamos por Talla y Temporada
                    tallas_sumadas = (
                        df_ventas_temp.groupby(['Talla', 'Temporada'])['Cantidad']
                        .sum()
                        .reset_index()
                    )
                    
                    tallas_en_agrupado = tallas_sumadas['Talla'].unique()
                   
                    # Verificar si hay datos válidos para el gráfico
                    if not tallas_sumadas.empty and len(tallas_sumadas) > 0:
                        # Orden personalizado de tallas
                        tallas_presentes = df_ventas_temp['Talla'].dropna().unique()
                        
                        if len(tallas_presentes) > 0:
                            try:
                                tallas_orden = sorted(tallas_presentes, key=custom_sort_key)
                                
        
                                
                                
                                # Asegurar que todas las tallas aparezcan en el gráfico
                                # Crear un DataFrame completo con todas las combinaciones talla-temporada
                                temporadas_unicas = df_ventas_temp['Temporada'].unique()
                                tallas_completas = []
                                
                                for talla in tallas_orden:
                                    for temporada in temporadas_unicas:
                                        # Buscar si existe esta combinación
                                        existing_data = tallas_sumadas[
                                            (tallas_sumadas['Talla'] == talla) & 
                                            (tallas_sumadas['Temporada'] == temporada)
                                        ]
                                        
                                        if not existing_data.empty:
                                            tallas_completas.append({
                                                'Talla': talla,
                                                'Temporada': temporada,
                                                'Cantidad': existing_data.iloc[0]['Cantidad']
                                            })
                                        else:
                                            # Agregar con cantidad 0 si no existe
                                            tallas_completas.append({
                                                'Talla': talla,
                                                'Temporada': temporada,
                                                'Cantidad': 0
                                            })
                                
                                # Crear DataFrame completo
                                tallas_sumadas_completo = pd.DataFrame(tallas_completas)
                                
                                # Verificar tallas en DataFrame completo
                                tallas_en_completo = tallas_sumadas_completo['Talla'].unique()
                                
                                
                                # Verificar cantidades por talla
                                cantidades_por_talla = tallas_sumadas_completo.groupby('Talla')['Cantidad'].sum()
                                
                                # Gráfico de barras apiladas por Temporada
                                temporada_colors = get_temporada_colors(df_ventas_temp)
                                
                                # Calcular altura dinámica basada en la cantidad de tallas
                                num_tallas = len(tallas_en_completo)  # Usar tallas del DataFrame completo
                                altura_dinamica = max(400, min(800, num_tallas * 50))  # Entre 400 y 800px
                                
                                # Ordenar tallas del DataFrame completo usando custom_sort_key
                                tallas_orden_completo = sorted(tallas_en_completo, key=custom_sort_key)
                                
                                
                                # Forzar todas las tallas a string para evitar problemas de categorías mixtas
                                tallas_sumadas_completo['Talla'] = tallas_sumadas_completo['Talla'].astype(str)
                                tallas_orden_completo = [str(t) for t in tallas_orden_completo]
                                
                                # Separar tallas numéricas y de letras
                                def es_numero(t):
                                    try:
                                        int(t)
                                        return True
                                    except:
                                        return False
                                
                                tallas_numericas = [t for t in tallas_orden_completo if es_numero(t)]
                                tallas_letras = [t for t in tallas_orden_completo if not es_numero(t)]
                                
                                # Filtrar DataFrame para cada tipo
                                df_num = tallas_sumadas_completo[tallas_sumadas_completo['Talla'].isin(tallas_numericas)]
                                df_let = tallas_sumadas_completo[tallas_sumadas_completo['Talla'].isin(tallas_letras)]
                                
                                # Mostrar gráfico de tallas numéricas si existen
                                if len(tallas_numericas) > 0 and not df_num.empty:
                                    altura_num = max(400, min(800, len(tallas_numericas) * 50))
                                    fig_num = px.bar(
                                        df_num,
                                        x='Talla',
                                        y='Cantidad',
                                        color='Temporada',
                                        text='Cantidad',
                                        category_orders={'Talla': tallas_numericas},
                                        color_discrete_map=temporada_colors,
                                        height=altura_num
                                    )
                                    max_cantidad = df_num['Cantidad'].max()
                                    y_max = max_cantidad * 1.1 if max_cantidad > 0 else 100
                                    fig_num.update_layout(
                                        xaxis_title="Talla",
                                        yaxis_title="Unidades Vendidas",
                                        barmode="stack",
                                        margin=dict(t=30, b=0, l=0, r=0),
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(0,0,0,0)",
                                        yaxis=dict(
                                            range=[0, y_max],
                                            showgrid=True,
                                            gridcolor='rgba(0,0,0,0.1)'
                                        ),
                                        xaxis=dict(
                                            categoryorder='array',
                                            categoryarray=tallas_numericas,
                                            showticklabels=True
                                        )
                                    )
                                    fig_num.update_traces(texttemplate='%{text:.0f}', textposition='inside', opacity=0.9)
                                    st.plotly_chart(fig_num, use_container_width=True)
                                
                                # Mostrar gráfico de tallas de letras si existen
                                if len(tallas_letras) > 0 and not df_let.empty:
                                    altura_let = max(400, min(800, len(tallas_letras) * 50))
                                    fig_let = px.bar(
                                        df_let,
                                        x='Talla',
                                        y='Cantidad',
                                        color='Temporada',
                                        text='Cantidad',
                                        category_orders={'Talla': tallas_letras},
                                        color_discrete_map=temporada_colors,
                                        height=altura_let
                                    )
                                    max_cantidad_let = df_let['Cantidad'].max()
                                    y_max_let = max_cantidad_let * 1.1 if max_cantidad_let > 0 else 100
                                    fig_let.update_layout(
                                        xaxis_title="Talla",
                                        yaxis_title="Unidades Vendidas",
                                        barmode="stack",
                                        margin=dict(t=30, b=0, l=0, r=0),
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(0,0,0,0)",
                                        yaxis=dict(
                                            range=[0, y_max_let],
                                            showgrid=True,
                                            gridcolor='rgba(0,0,0,0.1)'
                                        ),
                                        xaxis=dict(
                                            categoryorder='array',
                                            categoryarray=tallas_letras,
                                            showticklabels=True
                                        )
                                    )
                                    fig_let.update_traces(texttemplate='%{text:.0f}', textposition='inside', opacity=0.9)
                                    st.plotly_chart(fig_let, use_container_width=True)
                                
                                # Si no hay tallas válidas, mostrar advertencia
                                if (len(tallas_numericas) == 0 or df_num.empty) and (len(tallas_letras) == 0 or df_let.empty):
                                    st.warning("⚠️ No hay tallas válidas en los datos de ventas.")
                            except Exception as e:
                                st.warning(f"⚠️ Error al crear el gráfico de tallas: {str(e)}")
                                st.info("📊 Datos disponibles: " + str(tallas_sumadas_completo.head()))
                        else:
                            st.warning("⚠️ No hay tallas válidas en los datos de ventas.")
                    else:
                        st.warning("⚠️ No hay datos de ventas para mostrar el gráfico de tallas.")



            # Col 5,6: Tablas por Temporada con layout dinámico
            # Preparar datos de entrada en almacén para las tablas por temporada
            # Agregar Familia a df_productos usando Código único codes de df_ventas
            df_productos_temp = df_productos.copy()
            df_productos_temp['Fecha almacén'] = pd.to_datetime(df_productos_temp['Fecha almacén'], format='%d/%m/%Y', errors='coerce')

            # OPTIMIZACIÓN: Merge más eficiente con validación previa
            if 'Código único' in df_ventas.columns and 'Familia' in df_ventas.columns:
                
                
                # OPTIMIZACIÓN: Usar merge directo sin debug excesivo
                df_productos_temp = df_productos_temp.merge(
                    df_ventas[['Código único', 'Familia']].drop_duplicates(),
                    on='Código único',
                    how='left'
                )
                
                # Fill missing Familia values
                df_productos_temp['Familia'] = df_productos_temp['Familia'].fillna('Sin Familia')
            else:
                st.warning("⚠️ No se encontraron columnas 'Código único' o 'Familia' en df_ventas")
                df_productos_temp['Familia'] = 'Sin Familia'
            
            # OPTIMIZACIÓN: Filtrar por familia una sola vez
            # Obtener la familia más común en los datos filtrados
            familia_actual = df_ventas['Familia'].mode().iloc[0] if not df_ventas.empty else 'Sin Familia'
            df_almacen_fam = df_productos_temp[df_productos_temp['Familia'] == familia_actual].copy()
            
            
            # Filtrar por la familia actual
            df_almacen_fam = df_productos_temp[
                df_productos_temp['Familia'] == familia_actual
            ].copy()
            
           
            
            # Identificar productos sin familia asignada
            df_sin_familia = df_productos_temp[
                df_productos_temp['Familia'].isna()
            ].copy()
            
            # Inicializar df_pendientes como DataFrame vacío
            df_pendientes = pd.DataFrame()
            
            
            if not df_almacen_fam.empty and 'Fecha almacén' in df_almacen_fam.columns:
                
            
 
                # Separar filas con fecha válida y sin fecha
                df_almacen_fam_con_fecha = df_almacen_fam.dropna(subset=['Fecha almacén'])
                df_almacen_fam_sin_fecha = df_almacen_fam[df_almacen_fam['Fecha almacén'].isna()].copy()
                
                # Agregar mes de entrada para filas con fecha válida
                df_almacen_fam_con_fecha['Mes Entrada'] = df_almacen_fam_con_fecha['Fecha almacén'].dt.to_period('M').astype(str)
                
                # Separar filas pendientes de entrega (sin fecha válida)
                if not df_almacen_fam_sin_fecha.empty:
                    # Crear DataFrame separado para pendientes de entrega
                    df_pendientes = df_almacen_fam_sin_fecha.copy()
                    df_pendientes['Estado'] = 'Pendiente de entrega'
                else:
                    df_pendientes = pd.DataFrame()
                
                # Usar solo las filas con fecha válida para el análisis de almacén
                df_almacen_fam = df_almacen_fam_con_fecha
                
                # Obtener el último mes de df_ventas para filtrar los datos
                ultimo_mes_ventas = df_ventas['Mes'].max()
                
                # Preparar datos para la tabla por Temporada
                # Buscar la columna correcta para cantidad de entrada en almacén
            
                
                datos_tabla = (
                    df_almacen_fam.groupby(['Mes Entrada', 'Talla'])['Cantidad pedida']
                    .sum()
                    .reset_index()
                    .rename(columns={'Cantidad pedida': 'Cantidad Entrada Almacén'})
                    .sort_values(['Mes Entrada', 'Talla'])
                )
                
                # Filtrar datos hasta el último mes de ventas
                datos_tabla = datos_tabla[datos_tabla['Mes Entrada'] <= ultimo_mes_ventas]
                
                if not datos_tabla.empty:
                
                    # Obtener todos los temas únicos de df_productos (no solo los vendidos)
                    temas_productos = sorted(df_almacen_fam['Tema_temporada'].unique())
                    
                    # Calcular temas y num_temas SIEMPRE
                    temas = temas_productos
                    num_temas = len(temas)
                    

                    if num_temas > 0:
                        # --- Sección: Entradas almacén y traspasos ---
                        st.markdown('<hr style="margin: 1em 0; border-top: 2px solid #bbb;">', unsafe_allow_html=True)
                        st.markdown('<h4 style="color:#333;font-weight:bold;">Entradas almacén y traspasos</h4>', unsafe_allow_html=True)
                        
                        # Si se han seleccionado tiendas específicas, mostrar tabla de análisis temporal
                        
                        if tiendas_especificas:
                            st.subheader("Análisis Temporal: Entrada Almacén → Envío → Primera Venta")
                            # Preparar datos para el análisis temporal
                            df_almacen_fam_timeline = df_almacen_fam.copy()
                            df_traspasos_timeline = df_traspasos_filtrado.copy()
                            df_traspasos_timeline['Fecha enviado'] = pd.to_datetime(df_traspasos_timeline['Fecha enviado'], errors='coerce')
                            df_ventas_timeline = df_ventas.copy()
                            df_ventas_timeline['Fecha venta'] = pd.to_datetime(df_ventas_timeline['Fecha venta'], errors='coerce')

                            # 1. Solo el primer envío por tienda
                            df_traspasos_timeline = (
                                df_traspasos_timeline
                                .sort_values('Fecha enviado')
                                .drop_duplicates(subset=['Código único', 'Talla', 'Tienda'], keep='first')
                            )

                            # 2. Merge con almacén
                            merged = pd.merge(
                                df_almacen_fam_timeline,
                                df_traspasos_timeline,
                                left_on=['Código único', 'Talla'],
                                right_on=['Código único', 'Talla'],
                                suffixes=('_almacen', '_traspaso')
                            )
                            merged = merged[merged['Fecha enviado'] >= merged['Fecha almacén']]

                            timeline_data = []
                            for _, row in merged.iterrows():
                                fecha_entrada = row['Fecha almacén']
                                fecha_envio = row['Fecha enviado']
                                codigo_unico = row['Código único'].strip()
                                talla = row['Talla'].strip()
                                tienda_envio = row['Tienda']
                                tema = row['Tema']

                                # 3. Solo la primera venta en esa tienda para ese producto/talla
                                ventas_producto = df_ventas_timeline[
                                    (df_ventas_timeline['Código único'].str.strip() == codigo_unico) &
                                    (df_ventas_timeline['Talla'].str.strip() == talla) &
                                    (df_ventas_timeline['Tienda'].str.strip() == tienda_envio.strip()) &
                                    (df_ventas_timeline['Fecha venta'] >= fecha_entrada) &
                                    (df_ventas_timeline['Cantidad'] > 0)
                                ]
                                if not ventas_producto.empty:
                                    primera_venta = ventas_producto.sort_values('Fecha venta').iloc[0]
                                    fecha_primera_venta = primera_venta['Fecha venta']
                                    dias_entrada_venta = (fecha_primera_venta - fecha_envio).days
                                else:
                                    fecha_primera_venta = None
                                    dias_entrada_venta = -1
                                dias_entrada_envio = (fecha_envio - fecha_entrada).days
                                timeline_data.append({
                                    'Código único': codigo_unico,
                                    'Tema': tema,
                                    'Talla': talla,
                                    'Tienda Envío': tienda_envio,
                                    'Fecha Entrada Almacén': fecha_entrada.strftime('%d/%m/%Y'),
                                    'Fecha enviado a tienda': fecha_envio.strftime('%d/%m/%Y'),
                                    'Fecha Primera Venta': fecha_primera_venta.strftime('%d/%m/%Y') if fecha_primera_venta else "Sin ventas",
                                    'Días Entrada-Envío': dias_entrada_envio,
                                    'Días Envío-Primera Venta': dias_entrada_venta if dias_entrada_venta != -1 else -1
                                })

                            if timeline_data:
                                df_timeline = pd.DataFrame(timeline_data)
                                df_timeline['Fecha Entrada Almacén'] = pd.to_datetime(df_timeline['Fecha Entrada Almacén'], format='%d/%m/%Y')
                                df_timeline = df_timeline.sort_values('Fecha Entrada Almacén', ascending=False)
                                df_timeline['Fecha Entrada Almacén'] = df_timeline['Fecha Entrada Almacén'].dt.strftime('%d/%m/%Y')
                                st.dataframe(
                                    df_timeline,
                                    use_container_width=True,
                                    hide_index=True
                                )
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    avg_dias_envio = pd.to_numeric(df_timeline['Días Entrada-Envío'], errors='coerce').mean()
                                    st.metric("Promedio días Entrada→Envío", f"{avg_dias_envio:.1f} días")
                                with col2:
                                    avg_dias_venta = pd.to_numeric(df_timeline['Días Envío-Primera Venta'].replace('Sin ventas', pd.NA), errors='coerce').mean()
                                    st.metric("Promedio días Envío→Primera Venta", f"{avg_dias_venta:.1f} días" if not pd.isna(avg_dias_venta) else "N/A")
                                with col3:
                                    total_productos = len(df_timeline)
                                    st.metric("Total productos analizados", f"{total_productos}")
                            else:
                                st.info("No se encontraron datos de envíos para los productos de entrada en almacén de la familia seleccionada.")
                        
                        else:
                            if num_temas == 1:
                                # Un tema: centrado
                                col5a, col5b, col5c = st.columns([1, 2, 1])
                                with col5b:
                                    tema = temas[0]
                                    st.subheader(f"Entrada Almacén - {tema}")
                                    
                                    # Crear gráfico de comparación enviado vs ventas
                                    if tema == 'T_OI25':
                                        temporada_comparacion = 'I2025'
                                    elif tema == 'T_PV25':
                                        temporada_comparacion = 'V2025'
                                    elif tema == 'T_PV23':
                                        temporada_comparacion = 'V2023'
                                    elif tema == 'T_PV24':
                                        temporada_comparacion = 'V2024'
                                    elif tema == 'T_OI24':
                                        temporada_comparacion = 'I2024'
                                    elif tema == 'T_OI23':
                                        temporada_comparacion = 'I2023'
                                    else:
                                        temporada_comparacion = None
                                    
                                    if temporada_comparacion:
                                        ventas_temporada = df_ventas[df_ventas['Temporada'] == temporada_comparacion]
                                        if not ventas_temporada.empty:
                                            Código_único_tema = df_almacen_fam[df_almacen_fam['Tema_temporada'] == tema]['Código único'].unique()
                                            ventas_tema = ventas_temporada[ventas_temporada['Código único'].isin(Código_único_tema)]
                                            if not ventas_tema.empty:
                                                ventas_por_talla = ventas_tema.groupby('Talla')['Cantidad'].sum().reset_index()
                                                enviado_tema = df_almacen_fam[df_almacen_fam['Tema_temporada'] == tema]
                                                enviado_por_talla = enviado_tema.groupby('Talla')['Cantidad pedida'].sum().reset_index()
                                                datos_comparacion = pd.merge(
                                                    enviado_por_talla, 
                                                    ventas_por_talla, 
                                                    on='Talla', 
                                                    how='outer'
                                                ).fillna(0)
                                                # Ordenar tallas
                                                datos_comparacion = datos_comparacion.sort_values('Talla', key=lambda x: x.map(custom_sort_key))
                                                
                                                # Crear gráfico con plotly usando el mismo layout que "Unidades Vendidas por Talla"
                                                # Calcular altura dinámica basada en la cantidad de tallas
                                                num_tallas = len(datos_comparacion)
                                                altura_dinamica = max(400, min(800, num_tallas * 50))  # Entre 400 y 800px
                                                
                                                # Preparar datos para plotly
                                                datos_plotly = []
                                                for _, row in datos_comparacion.iterrows():
                                                    datos_plotly.append({
                                                        'Talla': row['Talla'],
                                                        'Tipo': 'Enviado Almacén',
                                                        'Cantidad': row['Cantidad pedida']
                                                    })
                                                    datos_plotly.append({
                                                        'Talla': row['Talla'],
                                                        'Tipo': 'Ventas',
                                                        'Cantidad': row['Cantidad']
                                                    })
                                                
                                                df_plotly = pd.DataFrame(datos_plotly)
                                                
                                                fig = px.bar(
                                                    df_plotly,
                                                    x='Talla',
                                                    y='Cantidad',
                                                    color='Tipo',
                                                    text='Cantidad',
                                                    category_orders={'Talla': datos_comparacion['Talla'].tolist()},
                                                    color_discrete_map={'Enviado Almacén': '#800080', 'Ventas': '#000080'},
                                                    height=altura_dinamica
                                                )
                                                
                                                # Calcular rango dinámico para el eje Y
                                                max_cantidad = df_plotly['Cantidad'].max()
                                                min_cantidad = df_plotly['Cantidad'].min()
                                                
                                                # Ajustar el rango del eje Y para que se vea bien
                                                if max_cantidad > 0:
                                                    # Agregar un 10% de margen arriba
                                                    y_max = max_cantidad * 1.1
                                                    # Para el mínimo, usar 0 o un valor pequeño
                                                    y_min = 0
                                                else:
                                                    y_max = 100
                                                    y_min = 0
                                                
                                                fig.update_layout(
                                                    title=f'Enviado vs Ventas - {tema} ({temporada_comparacion})',
                                                    xaxis_title="Talla",
                                                    yaxis_title="Cantidad",
                                                    barmode="group",
                                                    margin=dict(t=30, b=0, l=0, r=0),
                                                    paper_bgcolor="rgba(0,0,0,0)",
                                                    plot_bgcolor="rgba(0,0,0,0)",
                                                    yaxis=dict(
                                                        range=[y_min, y_max],
                                                        showgrid=True,
                                                        gridcolor='rgba(0,0,0,0.1)'
                                                    )
                                                )
                                                
                                                fig.update_traces(texttemplate='%{text:.0f}', textposition='inside', opacity=0.9)
                                                st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Filtrar datos para este tema específico
                                    datos_tema = df_almacen_fam[df_almacen_fam['Tema_temporada'] == tema]
                                    datos_tabla_tema = (
                                        datos_tema.groupby(['Mes Entrada', 'Talla'])['Cantidad pedida']
                                        .sum()
                                        .reset_index()
                                        .rename(columns={'Cantidad pedida': 'Cantidad Entrada Almacén'})
                                        .sort_values(['Mes Entrada', 'Talla'])
                                    )
                                    
                                    if not datos_tabla_tema.empty:
                                        # Crear tabla pivot para mejor visualización
                                        tabla_pivot = datos_tabla_tema.pivot_table(
                                            index='Mes Entrada',
                                            columns='Talla',
                                            values='Cantidad Entrada Almacén',
                                            fill_value=0
                                        ).round(0)
                                        tallas_orden = sorted(tabla_pivot.columns, key=custom_sort_key)
                                        tabla_pivot = tabla_pivot[tallas_orden]
                                        st.dataframe(
                                            tabla_pivot.style.format("{:,.0f}"),
                                            use_container_width=True,
                                            hide_index=False
                                        )
                                        total_temp = tabla_pivot.sum().sum()
                                        st.write(f"**Total Entrada Almacén:** {total_temp:,.0f}")
                                    else:
                                        st.info(f"No hay datos para el tema {tema}")
                            elif num_temas == 2:
                                col5, col6 = st.columns(2)
                                for i, tema in enumerate(temas):
                                    with locals()[f'col{5+i}']:
                                        st.subheader(f"Entrada Almacén - {tema}")
                                        
                                        # Crear gráfico de comparación enviado vs ventas
                                        if tema == 'T_OI25':
                                            temporada_comparacion = 'I2025'
                                        elif tema == 'T_PV25':
                                            temporada_comparacion = 'V2025'
                                        elif tema == 'T_PV23':
                                            temporada_comparacion = 'V2023'
                                        elif tema == 'T_PV24':
                                            temporada_comparacion = 'V2024'
                                        elif tema == 'T_OI24':
                                            temporada_comparacion = 'I2024'
                                        elif tema == 'T_OI23':
                                            temporada_comparacion = 'I2023'
                                        else:
                                            temporada_comparacion = None
                                        
                                        if temporada_comparacion:
                                            ventas_temporada = df_ventas[df_ventas['Temporada'] == temporada_comparacion]
                                            if not ventas_temporada.empty:
                                                Código_único_tema = df_almacen_fam[df_almacen_fam['Tema_temporada'] == tema]['Código único'].unique()
                                                ventas_tema = ventas_temporada[ventas_temporada['Código único'].isin(Código_único_tema)]
                                                if not ventas_tema.empty:
                                                    ventas_por_talla = ventas_tema.groupby('Talla')['Cantidad'].sum().reset_index()
                                                    enviado_tema = df_almacen_fam[df_almacen_fam['Tema_temporada'] == tema]
                                                    enviado_por_talla = enviado_tema.groupby('Talla')['Cantidad pedida'].sum().reset_index()
                                                    datos_comparacion = pd.merge(
                                                        enviado_por_talla, 
                                                        ventas_por_talla, 
                                                        on='Talla', 
                                                        how='outer'
                                                    ).fillna(0)
                                                    # Ordenar tallas
                                                    datos_comparacion = datos_comparacion.sort_values('Talla', key=lambda x: x.map(custom_sort_key))
                                                    
                                                    # Layout dinámico
                                                    num_tallas = len(datos_comparacion)
                                                    altura_dinamica = max(400, min(800, num_tallas * 50))  # Entre 400 y 800px
                                                    
                                                    # Preparar datos para plotly
                                                    datos_plotly = []
                                                    for _, row in datos_comparacion.iterrows():
                                                        datos_plotly.append({
                                                            'Talla': row['Talla'],
                                                            'Tipo': 'Enviado Almacén',
                                                            'Cantidad': row['Cantidad pedida']
                                                        })
                                                        datos_plotly.append({
                                                            'Talla': row['Talla'],
                                                            'Tipo': 'Ventas',
                                                            'Cantidad': row['Cantidad']
                                                        })
                                                    
                                                    df_plotly = pd.DataFrame(datos_plotly)
                                                    
                                                    fig = px.bar(
                                                        df_plotly,
                                                        x='Talla',
                                                        y='Cantidad',
                                                        color='Tipo',
                                                        text='Cantidad',
                                                        category_orders={'Talla': datos_comparacion['Talla'].tolist()},
                                                        color_discrete_map={'Enviado Almacén': '#800080', 'Ventas': '#000080'},
                                                        height=altura_dinamica
                                                    )
                                                    
                                                    # Rango dinámico eje Y
                                                    max_cantidad = df_plotly['Cantidad'].max()
                                                    y_max = max_cantidad * 1.1 if max_cantidad > 0 else 100
                                                    
                                                    fig.update_layout(
                                                        title=f'Enviado vs Ventas - {tema} ({temporada_comparacion})',
                                                        xaxis_title="Talla",
                                                        yaxis_title="Cantidad",
                                                        barmode="group",
                                                        margin=dict(t=30, b=0, l=0, r=0),
                                                        paper_bgcolor="rgba(0,0,0,0)",
                                                        plot_bgcolor="rgba(0,0,0,0)",
                                                        yaxis=dict(
                                                            range=[0, y_max],
                                                            showgrid=True,
                                                            gridcolor='rgba(0,0,0,0.1)'
                                                        )
                                                    )
                                                    fig.update_traces(texttemplate='%{text:.0f}', textposition='inside', opacity=0.9)
                                                    st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Filtrar datos para este tema específico
                                        datos_tema = df_almacen_fam[df_almacen_fam['Tema_temporada'] == tema]
                                        datos_tabla_tema = (
                                            datos_tema.groupby(['Mes Entrada', 'Talla'])['Cantidad pedida']
                                            .sum()
                                            .reset_index()
                                            .rename(columns={'Cantidad pedida': 'Cantidad Entrada Almacén'})
                                            .sort_values(['Mes Entrada', 'Talla'])
                                        )
                                        
                                        if not datos_tabla_tema.empty:
                                            # Crear tabla pivot para mejor visualización
                                            tabla_pivot = datos_tabla_tema.pivot_table(
                                                index='Mes Entrada',
                                                columns='Talla',
                                                values='Cantidad Entrada Almacén',
                                                fill_value=0
                                            ).round(0)
                                            tallas_orden = sorted(tabla_pivot.columns, key=custom_sort_key)
                                            tabla_pivot = tabla_pivot[tallas_orden]
                                            st.dataframe(
                                                tabla_pivot.style.format("{:,.0f}"),
                                                use_container_width=True,
                                                hide_index=False
                                            )
                                            total_temp = tabla_pivot.sum().sum()
                                            st.write(f"**Total Entrada Almacén:** {total_temp:,.0f}")
                                        else:
                                            st.info(f"No hay datos para el tema {tema}")
                            else:
                                col5, col6 = st.columns(2)
                                mitad = (num_temas + 1) // 2
                                temas_col5 = temas[:mitad]
                                temas_col6 = temas[mitad:]
                                with col5:
                                    for tema in temas_col5:
                                        st.subheader(f"Entrada Almacén - {tema}")
                                        
                                        # Crear gráfico de comparación enviado vs ventas
                                        if tema == 'T_OI25':
                                            temporada_comparacion = 'I2025'
                                        elif tema == 'T_PV25':
                                            temporada_comparacion = 'V2025'
                                        elif tema == 'T_PV23':
                                            temporada_comparacion = 'V2023'
                                        elif tema == 'T_PV24':
                                            temporada_comparacion = 'V2024'
                                        elif tema == 'T_OI24':
                                            temporada_comparacion = 'I2024'
                                        elif tema == 'T_OI23':
                                            temporada_comparacion = 'I2023'
                                        else:
                                            temporada_comparacion = None
                                        
                                        if temporada_comparacion:
                                            # Obtener datos de ventas para la temporada
                                            ventas_temporada = df_ventas[df_ventas['Temporada'] == temporada_comparacion]
                                            if not ventas_temporada.empty:
                                                # Obtener Código únicos del tema Código únicoual
                                                Código_único_tema = df_almacen_fam[df_almacen_fam['Tema_temporada'] == tema]['Código único'].unique()
                                                
                                                # Filtrar ventas por Código únicos del tema
                                                ventas_tema = ventas_temporada[ventas_temporada['Código único'].isin(Código_único_tema)]
                                                
                                                if not ventas_tema.empty:
                                                    # Agrupar ventas por talla
                                                    ventas_por_talla = ventas_tema.groupby('Talla')['Cantidad'].sum().reset_index()
                                                    
                                                    # Obtener datos de enviado del tema
                                                    enviado_tema = df_almacen_fam[df_almacen_fam['Tema_temporada'] == tema]
                                                    enviado_por_talla = enviado_tema.groupby('Talla')['Cantidad pedida'].sum().reset_index()
                                                    
                                                    # Combinar datos
                                                    datos_comparacion = pd.merge(
                                                        enviado_por_talla, 
                                                        ventas_por_talla, 
                                                        on='Talla', 
                                                        how='outer'
                                                    ).fillna(0)
                                                    
                                                    # Ordenar tallas
                                                    datos_comparacion = datos_comparacion.sort_values('Talla', key=lambda x: x.map(custom_sort_key))
                                                    
                                                    # Layout dinámico
                                                    num_tallas = len(datos_comparacion)
                                                    altura_dinamica = max(400, min(800, num_tallas * 50))  # Entre 400 y 800px
                                                    
                                                    # Preparar datos para plotly
                                                    datos_plotly = []
                                                    for _, row in datos_comparacion.iterrows():
                                                        datos_plotly.append({
                                                            'Talla': row['Talla'],
                                                            'Tipo': 'Enviado Almacén',
                                                            'Cantidad': row['Cantidad pedida']
                                                        })
                                                        datos_plotly.append({
                                                            'Talla': row['Talla'],
                                                            'Tipo': 'Ventas',
                                                            'Cantidad': row['Cantidad']
                                                        })
                                                    
                                                    df_plotly = pd.DataFrame(datos_plotly)
                                                    
                                                    fig = px.bar(
                                                        df_plotly,
                                                        x='Talla',
                                                        y='Cantidad',
                                                        color='Tipo',
                                                        text='Cantidad',
                                                        category_orders={'Talla': datos_comparacion['Talla'].tolist()},
                                                        color_discrete_map={'Enviado Almacén': '#800080', 'Ventas': '#000080'},
                                                        height=altura_dinamica
                                                    )
                                                    
                                                    # Rango dinámico eje Y
                                                    max_cantidad = df_plotly['Cantidad'].max()
                                                    y_max = max_cantidad * 1.1 if max_cantidad > 0 else 100
                                                    
                                                    fig.update_layout(
                                                        title=f'Enviado vs Ventas - {tema} ({temporada_comparacion})',
                                                        xaxis_title="Talla",
                                                        yaxis_title="Cantidad",
                                                        barmode="group",
                                                        margin=dict(t=30, b=0, l=0, r=0),
                                                        paper_bgcolor="rgba(0,0,0,0)",
                                                        plot_bgcolor="rgba(0,0,0,0)",
                                                        yaxis=dict(
                                                            range=[0, y_max],
                                                            showgrid=True,
                                                            gridcolor='rgba(0,0,0,0.1)'
                                                        )
                                                    )
                                                    fig.update_traces(texttemplate='%{text:.0f}', textposition='inside', opacity=0.9)
                                                    st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Filtrar datos para este tema específico
                                        datos_tema = df_almacen_fam[df_almacen_fam['Tema_temporada'] == tema]
                                        datos_tabla_tema = (
                                            datos_tema.groupby(['Mes Entrada', 'Talla'])['Cantidad pedida']
                                            .sum()
                                            .reset_index()
                                            .rename(columns={'Cantidad pedida': 'Cantidad Entrada Almacén'})
                                            .sort_values(['Mes Entrada', 'Talla'])
                                        )
                                        
                                        if not datos_tabla_tema.empty:
                                            # Crear tabla pivot para mejor visualización
                                            tabla_pivot = datos_tabla_tema.pivot_table(
                                                index='Mes Entrada',
                                                columns='Talla',
                                                values='Cantidad Entrada Almacén',
                                                fill_value=0
                                            ).round(0)
                                            tallas_orden = sorted(tabla_pivot.columns, key=custom_sort_key)
                                            tabla_pivot = tabla_pivot[tallas_orden]
                                            st.dataframe(
                                                tabla_pivot.style.format("{:,.0f}"),
                                                use_container_width=True,
                                                hide_index=False
                                            )
                                            total_temp = tabla_pivot.sum().sum()
                                            st.write(f"**Total Entrada Almacén:** {total_temp:,.0f}")
                                        else:
                                            st.info(f"No hay datos para el tema {tema}")
                                with col6:
                                    for tema in temas_col6:
                                        st.subheader(f"Entrada Almacén - {tema}")
                                        
                                        # Crear gráfico de comparación enviado vs ventas
                                        if tema == 'T_OI25':
                                            temporada_comparacion = 'I2025'
                                        elif tema == 'T_PV25':
                                            temporada_comparacion = 'V2025'
                                        elif tema == 'T_PV23':
                                            temporada_comparacion = 'V2023'
                                        elif tema == 'T_PV24':
                                            temporada_comparacion = 'V2024'
                                        elif tema == 'T_OI24':
                                            temporada_comparacion = 'I2024'
                                        elif tema == 'T_OI23':
                                            temporada_comparacion = 'I2023'
                                        else:
                                            temporada_comparacion = None
                                        
                                        if temporada_comparacion:
                                            # Obtener datos de ventas para la temporada
                                            ventas_temporada = df_ventas[df_ventas['Temporada'] == temporada_comparacion]
                                            if not ventas_temporada.empty:
                                                # Obtener Código únicos del tema Código únicoual
                                                Código_único_tema = df_almacen_fam[df_almacen_fam['Tema_temporada'] == tema]['Código único'].unique()
                                                
                                                # Filtrar ventas por Código únicos del tema
                                                ventas_tema = ventas_temporada[ventas_temporada['Código único'].isin(Código_único_tema)]
                                                
                                                if not ventas_tema.empty:
                                                    # Agrupar ventas por talla
                                                    ventas_por_talla = ventas_tema.groupby('Talla')['Cantidad'].sum().reset_index()
                                                    
                                                    # Obtener datos de enviado del tema
                                                    enviado_tema = df_almacen_fam[df_almacen_fam['Tema_temporada'] == tema]
                                                    enviado_por_talla = enviado_tema.groupby('Talla')['Cantidad pedida'].sum().reset_index()
                                                    
                                                    # Combinar datos
                                                    datos_comparacion = pd.merge(
                                                        enviado_por_talla, 
                                                        ventas_por_talla, 
                                                        on='Talla', 
                                                        how='outer'
                                                    ).fillna(0)
                                                    
                                                    # Ordenar tallas
                                                    datos_comparacion = datos_comparacion.sort_values('Talla', key=lambda x: x.map(custom_sort_key))
                                                    
                                                    # Layout dinámico
                                                    num_tallas = len(datos_comparacion)
                                                    altura_dinamica = max(400, min(800, num_tallas * 50))  # Entre 400 y 800px
                                                    
                                                    # Preparar datos para plotly
                                                    datos_plotly = []
                                                    for _, row in datos_comparacion.iterrows():
                                                        datos_plotly.append({
                                                            'Talla': row['Talla'],
                                                            'Tipo': 'Enviado Almacén',
                                                            'Cantidad': row['Cantidad pedida']
                                                        })
                                                        datos_plotly.append({
                                                            'Talla': row['Talla'],
                                                            'Tipo': 'Ventas',
                                                            'Cantidad': row['Cantidad']
                                                        })
                                                    
                                                    df_plotly = pd.DataFrame(datos_plotly)
                                                    
                                                    fig = px.bar(
                                                        df_plotly,
                                                        x='Talla',
                                                        y='Cantidad',
                                                        color='Tipo',
                                                        text='Cantidad',
                                                        category_orders={'Talla': datos_comparacion['Talla'].tolist()},
                                                        color_discrete_map={'Enviado Almacén': '#800080', 'Ventas': '#000080'},
                                                        height=altura_dinamica
                                                    )
                                                    
                                                    # Rango dinámico eje Y
                                                    max_cantidad = df_plotly['Cantidad'].max()
                                                    y_max = max_cantidad * 1.1 if max_cantidad > 0 else 100
                                                    
                                                    fig.update_layout(
                                                        title=f'Enviado vs Ventas - {tema} ({temporada_comparacion})',
                                                        xaxis_title="Talla",
                                                        yaxis_title="Cantidad",
                                                        barmode="group",
                                                        margin=dict(t=30, b=0, l=0, r=0),
                                                        paper_bgcolor="rgba(0,0,0,0)",
                                                        plot_bgcolor="rgba(0,0,0,0)",
                                                        yaxis=dict(
                                                            range=[0, y_max],
                                                            showgrid=True,
                                                            gridcolor='rgba(0,0,0,0.1)'
                                                        )
                                                    )
                                                    fig.update_traces(texttemplate='%{text:.0f}', textposition='inside', opacity=0.9)
                                                    st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Filtrar datos para este tema específico
                                        datos_tema = df_almacen_fam[df_almacen_fam['Tema_temporada'] == tema]
                                        datos_tabla_tema = (
                                            datos_tema.groupby(['Mes Entrada', 'Talla'])['Cantidad pedida']
                                            .sum()
                                            .reset_index()
                                            .rename(columns={'Cantidad pedida': 'Cantidad Entrada Almacén'})
                                            .sort_values(['Mes Entrada', 'Talla'])
                                        )
                                        
                                        if not datos_tabla_tema.empty:
                                            # Crear tabla pivot para mejor visualización
                                            tabla_pivot = datos_tabla_tema.pivot_table(
                                                index='Mes Entrada',
                                                columns='Talla',
                                                values='Cantidad Entrada Almacén',
                                                fill_value=0
                                            ).round(0)
                                            tallas_orden = sorted(tabla_pivot.columns, key=custom_sort_key)
                                            tabla_pivot = tabla_pivot[tallas_orden]
                                            st.dataframe(
                                                tabla_pivot.style.format("{:,.0f}"),
                                                use_container_width=True,
                                                hide_index=False
                                            )
                                            total_temp = tabla_pivot.sum().sum()
                                            st.write(f"**Total Entrada Almacén:** {total_temp:,.0f}")
                                        else:
                                            st.info(f"No hay datos para el tema {tema}")
                else:
                    st.info("No hay datos de entrada en almacén disponibles para la familia seleccionada.")

            # --- Tabla de Pendientes de Entrega ---
            if not df_pendientes.empty:
                st.markdown("---")
                viz_title("Pendientes de Entrega")
                
                # Preparar datos de pendientes por talla
                datos_pendientes = (
                    df_pendientes.groupby(['Talla'])['Cantidad pedida']
                    .sum()
                    .reset_index()
                    .rename(columns={'Cantidad pedida': 'Cantidad Pendiente'})
                    .sort_values('Talla', key=lambda x: x.map(custom_sort_key))
                )
                
                if not datos_pendientes.empty:
                    # Mostrar tabla de pendientes
                    st.dataframe(
                        datos_pendientes.style.format({
                            'Cantidad Pendiente': '{:,.0f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Mostrar total
                    total_pendientes = datos_pendientes['Cantidad Pendiente'].sum()
                    st.write(f"**Total Pendientes de Entrega:** {total_pendientes:,.0f}")
                    
                    # Mostrar información adicional
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total productos pendientes", len(df_pendientes))
                    with col2:
                        st.metric("Tallas diferentes", len(datos_pendientes))
                    with col3:
                        st.metric("Promedio por talla", f"{total_pendientes/len(datos_pendientes):,.0f}")
                else:
                    st.info("No hay datos de pendientes de entrega para mostrar.")
            else:
                pass


            # --- Tabla de Cantidad Pedida por Mes y Talla ---
            # Solo mostrar esta tabla cuando NO se han seleccionado tiendas específicas
            if not tiendas_especificas:
                st.markdown("---")
                viz_title("Cantidad Pedida por Mes y Talla")
                
                if not df_almacen_fam.empty and 'Cantidad pedida' in df_almacen_fam.columns:
                    # Preparar datos de cantidad pedida
                    datos_pedida = (
                        df_almacen_fam.groupby(['Mes Entrada', 'Talla'])['Cantidad pedida']
                        .sum()
                        .reset_index()
                        .rename(columns={'Mes Entrada': 'Mes', 'Cantidad pedida': 'Cantidad pedida'})
                        .sort_values(['Mes', 'Talla'])
                    )
                    
                    # Filtrar datos hasta el último mes de ventas
                    datos_pedida = datos_pedida[datos_pedida['Mes'] <= ultimo_mes_ventas]
                    
                    if not datos_pedida.empty:
                        # Crear tabla pivot para mejor visualización
                        tabla_pedida_pivot = datos_pedida.pivot_table(
                            index='Mes',
                            columns='Talla',
                            values='Cantidad pedida',
                            fill_value=0
                        ).round(0)
                        
                        # Ordenar tallas usando la función custom_sort_key
                        tallas_orden = sorted(tabla_pedida_pivot.columns, key=custom_sort_key)
                        tabla_pedida_pivot = tabla_pedida_pivot[tallas_orden]
                        
                        # Mostrar la tabla
                        st.dataframe(
                            tabla_pedida_pivot.style.format("{:,.0f}"),
                            use_container_width=True,
                            hide_index=False
                        )
                        
                        # Mostrar total
                        total_pedida = tabla_pedida_pivot.sum().sum()
                        st.write(f"**Total Cantidad Pedida:** {total_pedida:,.0f}")
                    else:
                        st.info("No hay datos de cantidad pedida para la familia seleccionada.")
                else:
                    st.info("No hay datos de cantidad pedida disponibles para la familia seleccionada.")

            # --- Ventas vs Traspasos por Tienda ---
            st.markdown("---")
            viz_title("Ventas vs Traspasos por Tienda")
            
            # Preparar datos de traspasos hasta la fecha máxima de ventas
            ultimo_mes_ventas = df_ventas['Mes'].max()
            df_traspasos_filtrado = df_traspasos_filtrado.copy()
            
            # Convertir fecha de traspasos y filtrar hasta el último mes de ventas
            df_traspasos_filtrado['Mes Enviado'] = pd.to_datetime(df_traspasos_filtrado['Fecha enviado']).dt.to_period('M').astype(str)
            df_traspasos_filtrado = df_traspasos_filtrado[df_traspasos_filtrado['Mes Enviado'] <= ultimo_mes_ventas]
            
            # Agrupar ventas por tienda y temporada
            ventas_por_tienda_temp = df_ventas.groupby(['Tienda', 'Temporada'])['Cantidad'].sum().reset_index()
            ventas_por_tienda_temp['Tipo'] = 'Ventas'
            ventas_por_tienda_temp = ventas_por_tienda_temp.rename(columns={'Cantidad': 'Cantidad Total'})
            
            # Obtener Código únicos que existen en ventas (limpiar espacios)
            Código_único_en_ventas = df_ventas['Código único'].str.strip().unique()
            
            
            # Filtrar traspasos para solo incluir Código únicos que están en ventas
            df_traspasos_filtrado_código_único = df_traspasos_filtrado[df_traspasos_filtrado['Código único'].isin(Código_único_en_ventas)]
            
            # Agrupar traspasos por tienda y temporada
            if not df_traspasos_filtrado_código_único.empty:
                # Asegurar que la columna Temporada existe en traspasos
                if 'Temporada' not in df_traspasos_filtrado_código_único.columns:
                    temporada_columns = [col for col in df_traspasos_filtrado_código_único.columns if 'temporada' in col.lower() or 'season' in col.lower()]
                    if temporada_columns:
                        df_traspasos_filtrado_código_único['Temporada'] = df_traspasos_filtrado_código_único[temporada_columns[0]]
                    else:
                        df_traspasos_filtrado_código_único['Temporada'] = 'Sin Temporada'
                else:
                    df_traspasos_filtrado_código_único['Temporada'] = df_traspasos_filtrado_código_único['Temporada'].fillna('Sin Temporada')
                
                # Limpiar temporada en traspasos para que coincida con ventas
                df_traspasos_filtrado_código_único['Temporada'] = df_traspasos_filtrado_código_único['Temporada'].str.strip().str[:5]
                
                traspasos_por_tienda_temp = df_traspasos_filtrado_código_único.groupby(['Tienda', 'Temporada'])['Cantidad enviada'].sum().reset_index()
                traspasos_por_tienda_temp['Tipo'] = 'Traspasos'
                traspasos_por_tienda_temp = traspasos_por_tienda_temp.rename(columns={'Cantidad enviada': 'Cantidad Total'})
            else:
                # Si no hay traspasos filtrados, crear un DataFrame vacío con la estructura correcta
                traspasos_por_tienda_temp = pd.DataFrame(columns=['Tienda', 'Temporada', 'Cantidad Total', 'Tipo'])
            
            # Combinar datos
            datos_comparacion = pd.concat([ventas_por_tienda_temp, traspasos_por_tienda_temp], ignore_index=True)
            
            if not datos_comparacion.empty:
                # Obtener top 20 tiendas por ventas totales
                top_tiendas_ventas = df_ventas.groupby('Tienda')['Cantidad'].sum().nlargest(50).index.tolist()
                
                # Filtrar datos para top 20 tiendas
                datos_top_tiendas = datos_comparacion[datos_comparacion['Tienda'].isin(top_tiendas_ventas)]
                
                if not datos_top_tiendas.empty:
                    # Crear gráfico con exCódigo únicoamente 2 barras por tienda (Ventas y Traspasos)
                    # Preparar datos para el nuevo formato
                    ventas_data = datos_top_tiendas[datos_top_tiendas['Tipo'] == 'Ventas'].copy()
                    traspasos_data = datos_top_tiendas[datos_top_tiendas['Tipo'] == 'Traspasos'].copy()
                    
                    # Obtener colores de temporada
                    temporada_colors = get_temporada_colors(df_ventas)
                    
                    # Crear figura
                    fig = go.Figure()
                    
                    # Obtener todas las tiendas únicas
                    tiendas_unicas = sorted(datos_top_tiendas['Tienda'].unique())
                    temporadas = sorted(datos_top_tiendas['Temporada'].unique())
                    
                    # Definir diferentes tonos de amarillo para traspasos por temporada
                    yellow_colors = ['#ffff00', '#ffeb3b', '#ffc107', '#ff9800', '#ff5722', '#f57c00', '#ef6c00', '#e65100']
                    
                    # Crear datos para cada tienda con dos barras (Ventas y Traspasos)
                    for tienda in tiendas_unicas:
                        # Datos de ventas para esta tienda
                        ventas_tienda = ventas_data[ventas_data['Tienda'] == tienda]
                        traspasos_tienda = traspasos_data[traspasos_data['Tienda'] == tienda]
                        
                        # Agregar barra de VENTAS (dividida por temporada)
                        if not ventas_tienda.empty:
                            for i, temporada in enumerate(temporadas):
                                ventas_temp = ventas_tienda[ventas_tienda['Temporada'] == temporada]
                                if not ventas_temp.empty:
                                    fig.add_trace(go.Bar(
                                        name=f'Ventas - {temporada}',
                                        x=[f'{tienda} - Ventas'],
                                        y=ventas_temp['Cantidad Total'],
                                        marker_color=temporada_colors.get(temporada, '#1f77b4'),
                                        text=ventas_temp['Cantidad Total'],
                                        texttemplate='%{text:,.0f}',
                                        textposition='inside',
                                        hovertemplate=f"Tienda: {tienda}<br>Tipo: Ventas<br>Temporada: {temporada}<br>Cantidad: %{{y:,.0f}}<extra></extra>",
                                        opacity=0.8,
                                        showlegend=True if tienda == tiendas_unicas[0] else False,  # Solo mostrar legend para la primera tienda
                                        legendgroup=f'Ventas - {temporada}'
                                    ))
                        
                        # Agregar barra de TRASPASOS (dividida por temporada)
                        if not traspasos_tienda.empty:
                            for i, temporada in enumerate(temporadas):
                                traspasos_temp = traspasos_tienda[traspasos_tienda['Temporada'] == temporada]
                                if not traspasos_temp.empty:
                                    # Usar diferentes tonos de amarillo para cada temporada
                                    yellow_color = yellow_colors[i % len(yellow_colors)]
                                    fig.add_trace(go.Bar(
                                        name=f'Traspasos - {temporada}',
                                        x=[f'{tienda} - Traspasos'],
                                        y=traspasos_temp['Cantidad Total'],
                                        marker_color=yellow_color,  # Diferentes tonos de amarillo por temporada
                                        text=traspasos_temp['Cantidad Total'],
                                        texttemplate='%{text:,.0f}',
                                        textposition='inside',
                                        hovertemplate=f"Tienda: {tienda}<br>Tipo: Traspasos<br>Temporada: {temporada}<br>Cantidad: %{{y:,.0f}}<extra></extra>",
                                        opacity=0.8,
                                        showlegend=True if tienda == tiendas_unicas[0] else False,  # Solo mostrar legend para la primera tienda
                                        legendgroup=f'Traspasos - {temporada}'
                                    ))
                    
                    # Configurar layout
                    fig.update_layout(
                        title="Ventas vs Traspasos por Tienda",
                        xaxis_title="Tienda",
                        yaxis_title="Cantidad Total",
                        barmode='stack',  # Barras apiladas por temporada
                        xaxis_tickangle=45,
                        showlegend=True,
                        margin=dict(t=30, b=0, l=0, r=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar tabla resumen con breakdown por temporada
                    st.subheader("Resumen de Ventas vs Traspasos por Temporada")
                    
                    # Tabla con breakdown por temporada
                    resumen_temporada = datos_top_tiendas.groupby(['Tienda', 'Tipo', 'Temporada'])['Cantidad Total'].sum().reset_index()
                    resumen_pivot_temp = resumen_temporada.pivot_table(
                        index=['Tienda', 'Temporada'], 
                        columns='Tipo', 
                        values='Cantidad Total', 
                        fill_value=0
                    ).reset_index()
                    
                    # Calcular totales por tienda
                    resumen_totales = datos_top_tiendas.groupby(['Tienda', 'Tipo'])['Cantidad Total'].sum().reset_index()
                    resumen_pivot_totales = resumen_totales.pivot(index='Tienda', columns='Tipo', values='Cantidad Total').fillna(0)
                    resumen_pivot_totales['Diferencia'] = resumen_pivot_totales['Ventas'] - resumen_pivot_totales['Traspasos']
                    
                    
                    resumen_pivot_totales['Eficiencia %'] = (resumen_pivot_totales['Ventas'] / resumen_pivot_totales['Traspasos'] * 100).fillna(0)

                    # Calcular Devoluciones (cantidad negativa) por tienda
                    devoluciones_por_tienda = df_ventas[df_ventas['Cantidad'] < 0].groupby('Tienda')['Cantidad'].sum().abs()
                    resumen_pivot_totales['Devoluciones'] = devoluciones_por_tienda.reindex(resumen_pivot_totales.index).fillna(0)
                    
                    # Calcular Ratio de devolución (Devoluciones / Ventas * 100)
                    resumen_pivot_totales['Ratio de devolución %'] = (resumen_pivot_totales['Devoluciones'] / resumen_pivot_totales['Ventas'] * 100).fillna(0)
                    
                    resumen_pivot_totales = resumen_pivot_totales.round(2)
                    
                    # Mostrar tabla de totales
                    st.write("**Totales por Tienda:**")
                    st.dataframe(
                        resumen_pivot_totales.style.format({
                            'Ventas': '{:,.0f}',
                            'Traspasos': '{:,.0f}',
                            'Diferencia': '{:,.0f}',
                            'Devoluciones': '{:,.0f}',
                            'Eficiencia %': '{:.1f}%',
                            'Ratio de devolución %': '{:.1f}%'
                        }),
                        use_container_width=True
                    )
                    
                    # Mostrar tabla detallada por temporada
                    st.write("**Detalle por Temporada:**")
                    st.dataframe(
                        resumen_pivot_temp.style.format({
                            'Ventas': '{:,.0f}',
                            'Traspasos': '{:,.0f}'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No hay datos suficientes para mostrar la comparación.")
            else:
                st.info("No hay datos de traspasos disponibles para la comparación.")

            


        except Exception as e:
            st.error(f"Error al calcular KPIs: {e}")

    elif seccion == "Geográfico y Tiendas":
        # Preparar datos
        ventas_por_zona = df_ventas.groupby('Zona geográfica')['Cantidad'].sum().reset_index()
        ventas_por_tienda = df_ventas.groupby('Tienda')['Cantidad'].sum().reset_index()
        tiendas_por_zona = df_ventas[['Tienda', 'Zona geográfica']].drop_duplicates().groupby('Zona geográfica').count().reset_index()

        # 1. KPIs: Mejor y peor tienda por zona
        viz_title("KPIs por Zona - Mejor y Peor Tienda")
        
        try:
            # Calcular ventas por tienda y zona
            ventas_tienda_zona = df_ventas.groupby(['Zona geográfica', 'Tienda']).agg({
                'Cantidad': 'sum',
                'Beneficio': 'sum'
            }).reset_index()
            
            # Asegurar que las columnas numéricas son del tipo correcto
            ventas_tienda_zona['Cantidad'] = pd.to_numeric(ventas_tienda_zona['Cantidad'], errors='coerce').fillna(0)
            ventas_tienda_zona['Beneficio'] = pd.to_numeric(ventas_tienda_zona['Beneficio'], errors='coerce').fillna(0)
            
            # Asegurar que Zona geográfica es string
            ventas_tienda_zona['Zona geográfica'] = ventas_tienda_zona['Zona geográfica'].astype(str)
            
            # Calcular media de ventas por zona
            media_por_zona = ventas_tienda_zona.groupby('Zona geográfica')['Cantidad'].mean().reset_index()
            media_por_zona = media_por_zona.rename(columns={'Cantidad': 'Media_Zona'})
            
            # Unir con ventas por tienda
            ventas_tienda_zona = ventas_tienda_zona.merge(media_por_zona, on='Zona geográfica')
            
            # Calcular porcentaje vs media con manejo de división por cero
            ventas_tienda_zona['%_vs_Media'] = 0.0  # Default value
            mask = ventas_tienda_zona['Media_Zona'] > 0
            ventas_tienda_zona.loc[mask, '%_vs_Media'] = (
                (ventas_tienda_zona.loc[mask, 'Cantidad'] - ventas_tienda_zona.loc[mask, 'Media_Zona']) / 
                ventas_tienda_zona.loc[mask, 'Media_Zona'] * 100
            ).round(1)
            
            # Encontrar mejor y peor tienda por zona con manejo de errores
            mejores_tiendas = []
            peores_tiendas = []
            
            for zona in ventas_tienda_zona['Zona geográfica'].unique():
                zona_data = ventas_tienda_zona[ventas_tienda_zona['Zona geográfica'] == zona].copy()
                if not zona_data.empty and len(zona_data) > 0:
                    # Encontrar mejor tienda (máxima cantidad)
                    try:
                        mejor_idx = zona_data['Cantidad'].idxmax()
                        if pd.notna(mejor_idx) and mejor_idx in zona_data.index:
                            mejores_tiendas.append(zona_data.loc[mejor_idx].to_dict())
                    except:
                        pass
                    
                    # Encontrar peor tienda (mínima cantidad)
                    try:
                        peor_idx = zona_data['Cantidad'].idxmin()
                        if pd.notna(peor_idx) and peor_idx in zona_data.index:
                            peores_tiendas.append(zona_data.loc[peor_idx].to_dict())
                    except:
                        pass
            
            mejores_tiendas = pd.DataFrame(mejores_tiendas) if mejores_tiendas else pd.DataFrame()
            peores_tiendas = pd.DataFrame(peores_tiendas) if peores_tiendas else pd.DataFrame()
            
            # Mostrar KPIs en formato de tarjetas
            zonas = sorted([str(z) for z in df_ventas['Zona geográfica'].unique() if pd.notna(z)])
            
            for zona in zonas:
                mejor = mejores_tiendas[mejores_tiendas['Zona geográfica'] == zona] if not mejores_tiendas.empty else pd.DataFrame()
                peor = peores_tiendas[peores_tiendas['Zona geográfica'] == zona] if not peores_tiendas.empty else pd.DataFrame()
                
                if not mejor.empty and not peor.empty:
                    try:
                        st.markdown(f"""
                        <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: white;">
                            <div style="color: #666666; font-size: 16px; font-weight: 600; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #e5e7eb;">
                                {zona}
                            </div>
                            <div style="display: flex; justify-content: space-between; gap: 15px;">
                                <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                                    <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Mejor Tienda</p>
                                    <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{mejor.iloc[0]['Tienda']}</p>
                                    <p style="color: #059669; font-size: 14px; margin: 5px 0 0 0;">{mejor.iloc[0]['Cantidad']:,.0f} uds</p>
                                    <p style="color: #059669; font-size: 14px; margin: 0;">{mejor.iloc[0]['Beneficio']:,.2f}€</p>
                                </div>
                                <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                                    <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Peor Tienda</p>
                                    <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{peor.iloc[0]['Tienda']}</p>
                                    <p style="color: #dc2626; font-size: 14px; margin: 5px 0 0 0;">{peor.iloc[0]['Cantidad']:,.0f} uds</p>
                                    <p style="color: #dc2626; font-size: 14px; margin: 0;">{peor.iloc[0]['Beneficio']:,.2f}€</p>
                                    <p style="color: #dc2626; font-size: 12px; margin: 5px 0 0 0;">{peor.iloc[0]['%_vs_Media']}% vs media</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error al mostrar KPIs para {zona}: {str(e)}")
                else:
                    st.warning(f"No hay datos suficientes para mostrar KPIs de {zona}")
            
        except Exception as e:
            st.error(f"Error al procesar KPIs por zona: {str(e)}")
            st.info("Mostrando información básica de zonas...")
            
            # Fallback: mostrar información básica
            zonas_basicas = df_ventas.groupby('Zona geográfica')['Cantidad'].sum().reset_index()
            st.dataframe(zonas_basicas, use_container_width=True)

        # 2. Row: Ventas por zona y Tiendas por zona
        col1, col2 = st.columns(2)
        
        with col1:
            viz_title("Ventas por Zona")
            fig = px.bar(ventas_por_zona, 
                        x='Zona geográfica', 
                        y='Cantidad',
                        color='Cantidad',
                        color_continuous_scale=COLOR_GRADIENT,
                        text='Cantidad')
            fig.update_layout(
                showlegend=False,
                xaxis_tickangle=45,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            fig.update_traces(opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            viz_title("Tiendas por Zona")
            fig = px.bar(tiendas_por_zona, 
                        x='Zona geográfica', 
                        y='Tienda',
                        color='Tienda',
                        color_continuous_scale=COLOR_GRADIENT,
                        text='Tienda')
            fig.update_layout(
                showlegend=False,
                xaxis_tickangle=45,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            fig.update_traces(opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)

        # 3. Row: Evolución mensual por zona
        viz_title("Evolución Mensual por Zona")
        zona_mes_evol = df_ventas.groupby(['Mes', 'Zona geográfica'])['Cantidad'].sum().reset_index()
        fig = px.line(zona_mes_evol, 
                     x='Mes', 
                     y='Cantidad',
                     color='Zona geográfica',
                     color_discrete_sequence=COLOR_GRADIENT)
        fig.update_layout(
            showlegend=True,
            legend_title_text='Zona Geográfica',
            xaxis_tickangle=45,
            margin=dict(t=30, b=0, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        fig.update_traces(opacity=0.8)
        st.plotly_chart(fig, use_container_width=True)

        # 4. Row: Mapa España y tabla
        col3, col4 = st.columns(2)
        
        with col3:
            viz_title("Mapa de Ventas - España")
            
            # Separar datos por país
            df_espana = df_ventas[~df_ventas['Tienda'].isin(TIENDAS_EXTRANJERAS)].copy()
            
            # Procesar datos de España usando Zona geográfica
            mapeo_zona_ciudad = {
                'Zona Madrid': 'MADRID',
                'Zona Andalucía': 'SEVILLA',
                'Zona Valencia': 'VALENCIA',
                'Zona Galicia': 'VIGO',
                'Zona Murcia': 'MURCIA',
                'Zona Castilla y León': 'SALAMANCA',
                'Zona País Vasco': 'BILBAO',
                'Zona Aragón': 'ZARAGOZA',
                'Zona Asturias': 'GIJON',
                'Zona Castilla-La Mancha': 'ALBACETE',
                'Zona Cataluña': 'BARCELONA',
                'Zona Cantabria': 'SANTANDER',
                'Zona Navarra': 'PAMPLONA',
                'Zona La Rioja': 'LOGROÑO',
                'Zona Extremadura': 'BADAJOZ',
                'Zona Canarias': 'LAS PALMAS',
                'Zona Baleares': 'PALMA'
            }
            
            # Asignar ciudad basada en zona geográfica
            df_espana['Ciudad'] = df_espana['Zona geográfica'].map(mapeo_zona_ciudad)
            
            # Para tiendas sin zona geográfica, intentar extraer del nombre
            df_espana['Ciudad'] = df_espana['Ciudad'].fillna(
                df_espana['Tienda'].str.extract(r'ET\d{1,2}-([\w\s\.\(\)]+)', expand=False)
                .str.upper()
                .str.replace(r'ECITRUCCO|ECI|XANADU|TRUCCO|CORT.*|\(.*\)', '', regex=True)
                .str.strip()
            )

            coordenadas_espana = {
                'MADRID': (40.4168, -3.7038),
                'SEVILLA': (37.3886, -5.9823),
                'MALAGA': (36.7213, -4.4214),
                'VALENCIA': (39.4699, -0.3763),
                'VIGO': (42.2406, -8.7207),
                'MURCIA': (37.9834, -1.1299),
                'SALAMANCA': (40.9701, -5.6635),
                'CORDOBA': (37.8882, -4.7794),
                'BILBAO': (43.2630, -2.9350),
                'ZARAGOZA': (41.6488, -0.8891),
                'JAEN': (37.7796, -3.7849),
                'GIJON': (43.5453, -5.6615),
                'ALBACETE': (38.9943, -1.8585),
                'GRANADA': (37.1773, -3.5986),
                'CARTAGENA': (37.6051, -0.9862),
                'TARRAGONA': (41.1189, 1.2445),
                'LEON': (42.5987, -5.5671),
                'SANTANDER': (43.4623, -3.8099),
                'PAMPLONA': (42.8125, -1.6458),
                'VITORIA': (42.8467, -2.6727),
                'CASTELLON': (39.9864, -0.0513),
                'CADIZ': (36.5271, -6.2886),
                'JEREZ': (36.6850, -6.1261),
                'AVILES': (43.5560, -5.9222),
                'BADAJOZ': (38.8794, -6.9707),
                'BARCELONA': (41.3851, 2.1734),
                'LOGROÑO': (42.4627, -2.4449),
                'LAS PALMAS': (28.1235, -15.4366),
                'PALMA': (39.5696, 2.6502)
            }

            # Procesar datos para España
            df_espana['lat'] = df_espana['Ciudad'].map(lambda c: coordenadas_espana.get(c, (None, None))[0])
            df_espana['lon'] = df_espana['Ciudad'].map(lambda c: coordenadas_espana.get(c, (None, None))[1])
            df_espana = df_espana.dropna(subset=['lat', 'lon'])

            # Agrupar por ciudad incluyendo tanto cantidad como ventas en euros
            ventas_ciudad_espana = df_espana.groupby(['Ciudad', 'lat', 'lon']).agg({
                'Cantidad': 'sum',
                'Beneficio': 'sum'
            }).reset_index()
            
            # --- FIX: asegurar que 'Cantidad' no tenga valores negativos ni NaN para el mapa ---
            ventas_ciudad_espana['Cantidad'] = pd.to_numeric(ventas_ciudad_espana['Cantidad'], errors='coerce').fillna(0)
            ventas_ciudad_espana['Cantidad'] = ventas_ciudad_espana['Cantidad'].clip(lower=0)
            # --- FIN FIX ---
            
            if not ventas_ciudad_espana.empty:
                fig_espana = px.scatter_mapbox(
                    ventas_ciudad_espana,
                    lat='lat',
                    lon='lon',
                    size='Cantidad',
                    color='Cantidad',
                    hover_name='Ciudad',
                    hover_data={'Cantidad': True, 'Beneficio': True},
                    color_continuous_scale='Viridis',
                    zoom=5,
                    height=400,
                    title="España - Ventas por Ciudad"
                )
                fig_espana.update_layout(
                    mapbox_style='open-street-map',
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_espana, use_container_width=True)
            else:
                st.info("No hay datos disponibles para España.")
        
        with col4:
            # Tabla de tiendas por ciudad - España
            if not ventas_ciudad_espana.empty:
                st.write("**Tiendas por Ciudad - España**")
                
                # Mostrar tabla resumida por ciudad con cantidad y euros
                resumen_espana = ventas_ciudad_espana.sort_values('Cantidad', ascending=False)
                st.dataframe(
                    resumen_espana[['Ciudad', 'Cantidad', 'Beneficio']].style.format({
                        'Cantidad': '{:,.0f}',
                        'Beneficio': '{:,.2f}€'
                    }),
                    use_container_width=True
                )
                
                
            else:
                st.info("No hay datos disponibles para España.")

        # 5. Row: Mapa Italia y tabla
        col5, col6 = st.columns(2)
        
        with col5:
            viz_title("Mapa de Ventas - Italia")
            
            # Separar datos por país
            df_italia = df_ventas[df_ventas['Tienda'].isin(TIENDAS_EXTRANJERAS)].copy()
            
            # Procesar datos de Italia
            df_italia['Ciudad'] = df_italia['Tienda'].str.extract(r'I\d{3}COIN([A-Z]+)', expand=False)

            coordenadas_italia = {
                'BERGAMO': (45.6983, 9.6773),
                'VARESE': (45.8206, 8.8256),
                'BARICASAMASSIMA': (40.9634, 16.7514),
                'MILANO5GIORNATE': (45.4642, 9.1900),
                'ROMACINECITTA': (41.9028, 12.4964),
                'GENOVA': (44.4056, 8.9463),
                'SASSARI': (40.7259, 8.5557),
                'CATANIA': (37.5079, 15.0830),
                'CAGLIARI': (39.2238, 9.1217),
                'LECCE': (40.3519, 18.1720),
                'MILANOCANTORE': (45.4642, 9.1900),
                'MESTRE': (45.4903, 12.2424),
                'PADOVA': (45.4064, 11.8768),
                'FIRENZE': (43.7696, 11.2558),
                'ROMASANGIOVANNI': (41.9028, 12.4964),
                'MILANO': (45.4642, 9.1900)
            }

            # Procesar datos para Italia
            df_italia['lat'] = df_italia['Ciudad'].map(lambda c: coordenadas_italia.get(c, (None, None))[0])
            df_italia['lon'] = df_italia['Ciudad'].map(lambda c: coordenadas_italia.get(c, (None, None))[1])
            df_italia = df_italia.dropna(subset=['lat', 'lon'])

            # Agrupar por ciudad incluyendo tanto cantidad como ventas en euros
            ventas_ciudad_italia = df_italia.groupby(['Ciudad', 'lat', 'lon']).agg({
                'Cantidad': 'sum',
                'Beneficio': 'sum'
            }).reset_index()
            
            # --- FIX: asegurar que 'Cantidad' no tenga valores negativos ni NaN para el mapa de Italia ---
            ventas_ciudad_italia['Cantidad'] = pd.to_numeric(ventas_ciudad_italia['Cantidad'], errors='coerce').fillna(0)
            ventas_ciudad_italia['Cantidad'] = ventas_ciudad_italia['Cantidad'].clip(lower=0)
            # --- FIN FIX ---
            
            if not ventas_ciudad_italia.empty:
                fig_italia = px.scatter_mapbox(
                    ventas_ciudad_italia,
                    lat='lat',
                    lon='lon',
                    size='Cantidad',
                    color='Cantidad',
                    hover_name='Ciudad',
                    hover_data={'Cantidad': True, 'Beneficio': True},
                    color_continuous_scale='Plasma',
                    zoom=5,
                    height=400,
                    title="Italia - Ventas por Ciudad"
                )
                fig_italia.update_layout(
                    mapbox_style='open-street-map',
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_italia, use_container_width=True)
            else:
                st.info("No hay datos disponibles para Italia.")
        
        with col6:
            # Tabla de tiendas por ciudad - Italia
            if not ventas_ciudad_italia.empty:
                st.write("**Tiendas por Ciudad - Italia**")
                
                # Mostrar tabla resumida por ciudad con cantidad y euros
                resumen_italia = ventas_ciudad_italia.sort_values('Cantidad', ascending=False)
                st.dataframe(
                    resumen_italia[['Ciudad', 'Cantidad', 'Beneficio']].style.format({
                        'Cantidad': '{:,.0f}',
                        'Beneficio': '{:,.2f}€'
                    }),
                    use_container_width=True
                )
                
                
            else:
                st.info("No hay datos disponibles para Italia.")


    elif seccion == "Producto, Campaña, Devoluciones y Rentabilidad":
        devoluciones = df_ventas[df_ventas['Cantidad'] < 0].copy()
        ventas = df_ventas[df_ventas['Cantidad'] > 0].copy()

        # Calcular descuento real basado en la diferencia entre PVP y precio real de venta
        if all(col in df_ventas.columns for col in ['PVP', 'Beneficio', 'Cantidad']):
            df_ventas['Precio Real Unitario'] = df_ventas['Beneficio'] / df_ventas['Cantidad']
            df_ventas['Descuento Real %'] = ((df_ventas['PVP'] - df_ventas['Precio Real Unitario']) / df_ventas['PVP'] * 100).fillna(0)
            df_ventas['Descuento Real %'] = df_ventas['Descuento Real %'].clip(0, 100)  # Limitar entre 0 y 100%

        # ===== KPIs =====
        st.markdown("### 📊 **KPIs de Devoluciones, Rebajas y Margen**")
        
        # Calculate KPIs
        # Tienda con más devoluciones y ratio
        tienda_mas_devoluciones = "Sin datos"
        ratio_devolucion_valor = 0
        if not devoluciones.empty:
            devoluciones_por_tienda = devoluciones.groupby('Tienda').agg({'Cantidad': 'sum'}).reset_index()
            devoluciones_por_tienda['Cantidad'] = abs(devoluciones_por_tienda['Cantidad'])
            devoluciones_por_tienda = devoluciones_por_tienda.sort_values('Cantidad', ascending=False)
            
            # Calcular ratio de devolución por tienda
            ventas_por_tienda = ventas.groupby('Tienda')['Cantidad'].sum().reset_index()
            ratio_devolucion = ventas_por_tienda.merge(devoluciones_por_tienda, on='Tienda', how='left')
            ratio_devolucion['Cantidad_y'] = ratio_devolucion['Cantidad_y'].fillna(0)
            ratio_devolucion['Ratio Devolución %'] = (ratio_devolucion['Cantidad_y'] / ratio_devolucion['Cantidad_x'] * 100).round(2)
            
            top_tienda_devolucion = ratio_devolucion.loc[ratio_devolucion['Cantidad_y'].idxmax()]
            tienda_mas_devoluciones = top_tienda_devolucion['Tienda']
            ratio_devolucion_valor = top_tienda_devolucion['Ratio Devolución %']
        
        # Talla más devuelta
        talla_mas_devuelta = "Sin datos"
        talla_devuelta_unidades = 0
        if not devoluciones.empty and 'Talla' in devoluciones.columns:
            talla_mas_devuelta_data = devoluciones.groupby('Talla')['Cantidad'].sum().abs().sort_values(ascending=False).head(1)
            if not talla_mas_devuelta_data.empty:
                talla_mas_devuelta = talla_mas_devuelta_data.index[0]
                talla_devuelta_unidades = talla_mas_devuelta_data.iloc[0]
        
        # Familia más devuelta (excluyendo GR.ART.FICTICIO)
        familia_mas_devuelta = "Sin datos"
        familia_devuelta_unidades = 0
        familia_ficticio_unidades = 0
        if not devoluciones.empty:
            # Excluir 'GR.ART.FICTICIO' para el ranking principal
            devoluciones_sin_ficticio = devoluciones[devoluciones['Familia'] != 'GR.ART.FICTICIO']
            familia_mas_devuelta_data = devoluciones_sin_ficticio.groupby('Familia')['Cantidad'].sum().abs().sort_values(ascending=False).head(1)
            if not familia_mas_devuelta_data.empty:
                familia_mas_devuelta = familia_mas_devuelta_data.index[0]
                familia_devuelta_unidades = familia_mas_devuelta_data.iloc[0]
            # Calcular valor para 'GR.ART.FICTICIO' si existe
            ficticio_data = devoluciones[devoluciones['Familia'] == 'GR.ART.FICTICIO'].groupby('Familia')['Cantidad'].sum().abs()
            if not ficticio_data.empty:
                familia_ficticio_unidades = ficticio_data.iloc[0]
        
        # Rebajas 1ª (Enero y Junio)
        ventas_rebajas_1 = 0
        porcentaje_rebajas_1 = 0
        if 'Fecha venta' in df_ventas.columns:
            df_ventas_temp = df_ventas.copy()
            df_ventas_temp['Fecha venta'] = pd.to_datetime(df_ventas_temp['Fecha venta'], errors='coerce')
            df_ventas_temp['mes'] = df_ventas_temp['Fecha venta'].dt.month
            df_ventas_temp = df_ventas_temp[df_ventas_temp['Beneficio'] > 0]
            rebajas_1 = df_ventas_temp[df_ventas_temp['mes'].isin([1, 6])]
            
            if 'precio_pvp' in rebajas_1.columns:
                rebajas_1['Precio_venta'] = rebajas_1['Beneficio'] / rebajas_1['Cantidad']
                rebajas_1_real = rebajas_1[rebajas_1['Precio_venta'] < rebajas_1['precio_pvp']]
                ventas_rebajas_1 = rebajas_1_real['Beneficio'].sum()
            else:
                ventas_rebajas_1 = rebajas_1['Beneficio'].sum()
            
            if df_ventas_temp['Beneficio'].sum() > 0:
                porcentaje_rebajas_1 = (ventas_rebajas_1 / df_ventas_temp['Beneficio'].sum() * 100)
        
        # Rebajas 2ª (Febrero y Julio)
        ventas_rebajas_2 = 0
        porcentaje_rebajas_2 = 0
        if 'Fecha venta' in df_ventas.columns:
            df_ventas_temp = df_ventas.copy()
            df_ventas_temp['Fecha venta'] = pd.to_datetime(df_ventas_temp['Fecha venta'], errors='coerce')
            df_ventas_temp['mes'] = df_ventas_temp['Fecha venta'].dt.month
            df_ventas_temp = df_ventas_temp[df_ventas_temp['Beneficio'] > 0]
            rebajas_2 = df_ventas_temp[df_ventas_temp['mes'].isin([2, 7])]
            
            if 'precio_pvp' in rebajas_2.columns:
                rebajas_2['Precio_venta'] = rebajas_2['Beneficio'] / rebajas_2['Cantidad']
                rebajas_2_real = rebajas_2[rebajas_2['Precio_venta'] < rebajas_2['precio_pvp']]
                ventas_rebajas_2 = rebajas_2_real['Beneficio'].sum()
            else:
                ventas_rebajas_2 = rebajas_2['Beneficio'].sum()
            
            if df_ventas_temp['Beneficio'].sum() > 0:
                porcentaje_rebajas_2 = (ventas_rebajas_2 / df_ventas_temp['Beneficio'].sum() * 100)
        
        # Margen bruto por unidad (promedio)
        
        margen_unitario_promedio = 0
        margen_unitario_promedio_positivo = 0
        if all(col in df_ventas.columns for col in ['PVP', 'Precio Coste']):
            df_ventas_temp = df_ventas.copy()
            df_ventas_temp['margen_unitario'] = df_ventas_temp['PVP'] - df_ventas_temp['Precio Coste']
            margen_unitario_promedio = df_ventas_temp['margen_unitario'].mean()
            margen_unitario_promedio_positivo = df_ventas_temp[df_ventas_temp['margen_unitario'] > 0]['margen_unitario'].mean()
        
        # Margen porcentual (promedio)
        margen_porcentual_promedio = 0
        margen_porcentual_promedio_positivo = 0
        if all(col in df_ventas.columns for col in ['PVP', 'Precio Coste']):
            df_ventas_temp = df_ventas.copy()
            df_ventas_temp['margen_unitario'] = df_ventas_temp['PVP'] - df_ventas_temp['Precio Coste']
            df_ventas_temp['margen_%'] = df_ventas_temp['margen_unitario'] / df_ventas_temp['PVP']
            # Ignorar productos con PVP = 0 para el promedio real
            df_ventas_temp_validos = df_ventas_temp[df_ventas_temp['PVP'] != 0]
            margen_porcentual_promedio = df_ventas_temp_validos['margen_%'].mean() * 100
            margen_porcentual_promedio_positivo = df_ventas_temp_validos[df_ventas_temp_validos['margen_%'] > 0]['margen_%'].mean() * 100
        
        # KPIs in HTML style like Resumen General
        st.markdown("""
            <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: white;">
                <div style="color: #666666; font-size: 16px; font-weight: 600; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #e5e7eb;">
                    KPIs de Devoluciones y Rebajas
                </div>
                <div style="display: flex; justify-content: space-between; gap: 15px;">
                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Tienda con más devoluciones</p>
                        <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{}</p>
                        <p style="color: #dc2626; font-size: 12px; margin: 0;">Ratio: {:.1f}%</p>
                    </div>
                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Talla más devuelta</p>
                        <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{}</p>
                        <p style="color: #dc2626; font-size: 12px; margin: 0;">{:.0f} unidades</p>
                    </div>
                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Familia más devuelta (sin ficticio)</p>
                        <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{}</p>
                        <p style="color: #dc2626; font-size: 12px; margin: 0;">{:.0f} unidades</p>
                        {} <!-- Aquí mostramos el ficticio si existe -->
                    </div>
                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Rebajas 1ª (Enero/Junio)</p>
                        <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{:,.0f}€</p>
                        <p style="color: #059669; font-size: 12px; margin: 0;">{:.1f}% del total</p>
                    </div>
                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Rebajas 2ª (Febrero/Julio)</p>
                        <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{:,.0f}€</p>
                        <p style="color: #059669; font-size: 12px; margin: 0;">{:.1f}% del total</p>
                    </div>
                </div>
            </div>
        """.format(
            tienda_mas_devoluciones, ratio_devolucion_valor, talla_mas_devuelta, talla_devuelta_unidades, 
            familia_mas_devuelta, familia_devuelta_unidades,
            f'<p style="color: #666666; font-size: 12px; margin: 0;">GR.ART.FICTICIO: {familia_ficticio_unidades:.0f} unidades</p>' if familia_ficticio_unidades > 0 else '',
            ventas_rebajas_1, porcentaje_rebajas_1, 
            ventas_rebajas_2, porcentaje_rebajas_2), unsafe_allow_html=True)
        
        # Margen KPIs in separate row
        st.markdown("""
            <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: white;">
                <div style="color: #666666; font-size: 16px; font-weight: 600; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #e5e7eb;">
                    KPIs de Margen
                </div>
                <div style="display: flex; justify-content: space-between; gap: 15px;">
                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Margen Unitario Promedio Solo Positivo</p>
                        <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{:.2f}€</p>
                        <p style="color: #059669; font-size: 12px; margin: 0;">por unidad (solo positivos)</p>
                    </div>
                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Margen % Promedio Solo Positivo</p>
                        <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{:.1f}%</p>
                        <p style="color: #059669; font-size: 12px; margin: 0;">del PVP (solo positivos)</p>
                    </div>
                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Margen Unitario Promedio Real</p>
                        <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{:.2f}€</p>
                        <p style="color: #059669; font-size: 12px; margin: 0;">por unidad (real)</p>
                    </div>
                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Margen % Promedio Real</p>
                        <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{:.1f}%</p>
                        <p style="color: #059669; font-size: 12px; margin: 0;">del PVP (real)</p>
                    </div>
                </div>
            </div>
        """.format(
            margen_unitario_promedio_positivo if margen_unitario_promedio_positivo is not None else 0,
            margen_porcentual_promedio_positivo if margen_porcentual_promedio_positivo is not None else 0,
            margen_unitario_promedio if margen_unitario_promedio is not None else 0,
            margen_porcentual_promedio if margen_porcentual_promedio is not None else 0
        ), unsafe_allow_html=True)

        # Tabla de depuración: productos con margen negativo
        if all(col in df_ventas.columns for col in ['PVP', 'Precio Coste']):
            df_ventas_temp = df_ventas.copy()
            df_ventas_temp['margen_unitario'] = df_ventas_temp['PVP'] - df_ventas_temp['Precio Coste']
            productos_margen_negativo = df_ventas_temp[df_ventas_temp['margen_unitario'] < 0]
            if not productos_margen_negativo.empty:
                st.markdown('### Tabla de depuración: Productos con margen negativo (PVP < Precio Coste)')
                st.dataframe(productos_margen_negativo[['Código único', 'Familia', 'Temporada', 'Fecha venta', 'PVP', 'Precio Coste', 'margen_unitario']], use_container_width=True, hide_index=True)

        # ===== GRÁFICOS =====
        st.markdown("### **Análisis de Devoluciones y Temporadas**")

        # Row 1: Ventas vs Devoluciones por Familia
        st.markdown("#### **Ventas vs Devoluciones por Familia**")
        
        if not devoluciones.empty:
            # Preparar datos para comparación
            ventas_por_familia = ventas.groupby('Familia')['Cantidad'].sum().reset_index()
            ventas_por_familia['Tipo'] = 'Ventas'
            
            devoluciones_por_familia = devoluciones.groupby('Familia')['Cantidad'].sum().reset_index()
            devoluciones_por_familia['Cantidad'] = abs(devoluciones_por_familia['Cantidad'])
            devoluciones_por_familia['Tipo'] = 'Devoluciones'
            
            # Combinar datos
            comparacion_familias = pd.concat([ventas_por_familia, devoluciones_por_familia], ignore_index=True)
            
            # Crear gráfico de barras agrupadas
            fig = px.bar(
                comparacion_familias,
                x='Familia',
                y='Cantidad',
                color='Tipo',
                color_discrete_map={'Ventas': '#0066cc', 'Devoluciones': '#ff4444'},
                barmode='group',
                title="Ventas vs Devoluciones por Familia"
            )
            
            fig.update_layout(
                xaxis_tickangle=45,
                height=500,
                showlegend=True,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos de devoluciones disponibles para mostrar la comparación.")

        # Row 2: Talla más devuelta y menos devuelta por familia
        st.markdown("#### **Análisis de Tallas por Familia**")
        
        if not devoluciones.empty and 'Talla' in devoluciones.columns:
            col_talla1, col_talla2 = st.columns(2)
            
            with col_talla1:
                # Talla más devuelta por familia
                talla_mas_devuelta_familia = devoluciones.groupby(['Familia', 'Talla'])['Cantidad'].sum().abs().reset_index()
                talla_mas_devuelta_familia = talla_mas_devuelta_familia.loc[talla_mas_devuelta_familia.groupby('Familia')['Cantidad'].idxmax()]
                
                fig = px.bar(
                    talla_mas_devuelta_familia,
                    x='Familia',
                    y='Cantidad',
                    color='Talla',
                    title="Talla más devuelta por Familia",
                    color_discrete_sequence=px.colors.sequential.Reds
                )
                fig.update_layout(
                    xaxis_tickangle=45,
                    height=400,
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_talla2:
                # Talla menos devuelta por familia
                talla_menos_devuelta_familia = devoluciones.groupby(['Familia', 'Talla'])['Cantidad'].sum().abs().reset_index()
                talla_menos_devuelta_familia = talla_menos_devuelta_familia.loc[talla_menos_devuelta_familia.groupby('Familia')['Cantidad'].idxmin()]
                
                fig = px.bar(
                    talla_menos_devuelta_familia,
                    x='Familia',
                    y='Cantidad',
                    color='Talla',
                    title="Talla menos devuelta por Familia",
                    color_discrete_sequence=px.colors.sequential.Reds
                )
                fig.update_layout(
                    xaxis_tickangle=45,
                    height=400,
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos de tallas en las devoluciones disponibles.")

        # Row 3: Análisis de ventas en/ fuera de temporada
        st.markdown("#### ** Análisis de Ventas por Temporada**")
        
        if 'Temporada' in df_ventas.columns:
            # Función para determinar si un producto fue vendido fuera de su temporada
            def vendido_fuera_temporada(row):
                temporada = row['Temporada']
                fecha = row['Fecha venta']

                # Si la temporada no está bien definida, marcamos como fuera de temporada
                if not isinstance(temporada, str) or len(temporada) < 5:
                    return 1

                tipo_temporada = temporada[0]   # 'I' o 'V'
                ano_temporada_str = temporada[1:]

                # Validar que ano_temporada_str es numérico y tipo_temporada es válido
                if tipo_temporada not in ['I', 'V'] or not ano_temporada_str.isdigit():
                    return 1

                ano_temporada = int(ano_temporada_str)

                if tipo_temporada == 'I':  # Invierno: sept (año-1) a feb (año)
                    inicio = pd.Timestamp(year=ano_temporada - 1, month=9, day=1)
                    fin = pd.Timestamp(year=ano_temporada, month=2, day=28)  # ignoramos bisiestos
                    if inicio <= fecha <= fin:
                        return 0  # Vendido dentro de su temporada (Invierno)
                    else:
                        return 1  # Vendido fuera de temporada

                elif tipo_temporada == 'V':  # Verano: marzo a agosto año
                    inicio = pd.Timestamp(year=ano_temporada, month=3, day=1)
                    fin = pd.Timestamp(year=ano_temporada, month=8, day=31)
                    if inicio <= fecha <= fin:
                        return 0  # Vendido dentro de su temporada (Verano)
                    else:
                        return 1  # Vendido fuera de temporada
                else:
                    return 1  # Temporada no reconocida

            # Aplicar la función al DataFrame
            df_ventas_temp = df_ventas.copy()
            df_ventas_temp['vendido_fuera_temporada'] = df_ventas_temp.apply(vendido_fuera_temporada, axis=1)
            
            # Agrupar por temporada y tipo de venta
            analisis_temporada = df_ventas_temp.groupby(['Temporada', 'vendido_fuera_temporada'])['Cantidad'].sum().reset_index()
            analisis_temporada['Tipo_Venta'] = analisis_temporada['vendido_fuera_temporada'].map({
                0: 'En Temporada',
                1: 'Fuera de Temporada'
            })
            
            # Crear gráfico
            fig = px.bar(
                analisis_temporada,
                x='Temporada',
                y='Cantidad',
                color='Tipo_Venta',
                color_discrete_map={'En Temporada': '#0066cc', 'Fuera de Temporada': '#ff4444'},
                barmode='stack',
                title="Ventas En vs Fuera de Temporada por Campaña"
            )
            
            fig.update_layout(
                xaxis_tickangle=45,
                height=500,
                showlegend=True,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar tabla resumen
            st.markdown("**Resumen de Ventas por Temporada:**")
            resumen_temporada = analisis_temporada.pivot_table(
                index='Temporada',
                columns='Tipo_Venta',
                values='Cantidad',
                fill_value=0
            ).reset_index()
            # Asegurar que ambas columnas existen
            for col in ['En Temporada', 'Fuera de Temporada']:
                if col not in resumen_temporada.columns:
                    resumen_temporada[col] = 0
            # Calcular porcentajes
            resumen_temporada['Total'] = resumen_temporada['En Temporada'] + resumen_temporada['Fuera de Temporada']
            resumen_temporada['% En Temporada'] = (resumen_temporada['En Temporada'] / resumen_temporada['Total'] * 100).round(1)
            resumen_temporada['% Fuera de Temporada'] = (resumen_temporada['Fuera de Temporada'] / resumen_temporada['Total'] * 100).round(1)
            # Ordenar columnas para visual consistencia
            resumen_temporada = resumen_temporada[['Temporada', 'En Temporada', 'Fuera de Temporada', 'Total', '% En Temporada', '% Fuera de Temporada']]
            st.dataframe(
                resumen_temporada,
                use_container_width=True,
                hide_index=True
            )
            
        else:
            st.info("No hay datos de temporada disponibles para el análisis.")

        # ===== TABLA DE PRODUCTOS CON BAJO MARGEN (al final) =====
        st.markdown("---")
        st.markdown("### **Productos con Bajo Margen**")
        
        # Buscar columnas que podrían ser Precio Coste
        columnas_disponibles = df_ventas.columns.tolist()
        coste_col = None
        
        # Buscar Precio Coste
        for col in ['precio_cost', 'precio_coste', 'Precio Coste', 'Precio Costo', 'Coste', 'Costo', 'coste', 'costo']:
            if col in columnas_disponibles:
                coste_col = col
                break
        
        if coste_col and 'Beneficio' in df_ventas.columns and 'Cantidad' in df_ventas.columns:
            # Slider para ajustar el umbral de margen
            umbral_margen = st.slider(
                "Umbral de margen % (productos por debajo de este valor):",
                min_value=0.0,
                max_value=1.0,
                value=0.36,
                step=0.01,
                format="%.2f"
            )
            
            # Calcular márgenes usando Beneficio (Beneficio) como precio de venta
            # Excluir devoluciones (Cantidad < 0)
            df_ventas_temp = df_ventas[df_ventas['Cantidad'] > 0].copy()
            df_ventas_temp['Precio_venta'] = df_ventas_temp['Beneficio'] / df_ventas_temp['Cantidad']
            df_ventas_temp['margen_unitario'] = df_ventas_temp['Precio_venta'] - df_ventas_temp[coste_col]
            df_ventas_temp['margen_%'] = df_ventas_temp['margen_unitario'] / df_ventas_temp['Precio_venta']
            
            # Filtrar productos con margen bajo (incluyendo márgenes negativos)
            productos_bajo_margen = df_ventas_temp[df_ventas_temp['margen_%'] < umbral_margen].copy()
            
            if not productos_bajo_margen.empty:
                # Preparar tabla con las columnas solicitadas
                tabla_bajo_margen = productos_bajo_margen[[
                    'Código único', 'Familia', 'Temporada', 'Fecha venta', 
                    'Precio_venta', coste_col, 'margen_'
                ]].copy()
                
                # Formatear columnas
                tabla_bajo_margen['Fecha venta'] = pd.to_datetime(tabla_bajo_margen['Fecha venta']).dt.strftime('%d/%m/%Y')
                tabla_bajo_margen['Precio_venta'] = tabla_bajo_margen['Precio_venta'].round(2)
                tabla_bajo_margen[coste_col] = tabla_bajo_margen[coste_col].round(2)
                tabla_bajo_margen['margen_%'] = (tabla_bajo_margen['margen_%'] * 100).round(1)
                
                # Renombrar columnas para mejor visualización
                tabla_bajo_margen.columns = [
                    'Código único', 'Familia', 'Temporada', 'Fecha Venta', 
                    'Precio Venta (€)', f'{coste_col} (€)', 'Margen %'
                ]
                
                st.markdown(f"**Productos con margen inferior al {umbral_margen*100:.0f}% ({len(tabla_bajo_margen)} productos):**")
                st.dataframe(
                    tabla_bajo_margen,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Estadísticas adicionales con manejo de errores
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Total productos", len(tabla_bajo_margen))
                with col_stats2:
                    margen_promedio_bajo = tabla_bajo_margen['Margen %'].mean()
                    if pd.isna(margen_promedio_bajo) or margen_promedio_bajo == float('inf') or margen_promedio_bajo == float('-inf'):
                        st.metric("Margen promedio", "N/A")
                    else:
                        st.metric("Margen promedio", f"{margen_promedio_bajo:.1f}%")
                with col_stats3:
                    # Calcular pérdida estimada de manera más robusta
                    try:
                        # Solo considerar productos con margen negativo o muy bajo
                        productos_perdida = tabla_bajo_margen[tabla_bajo_margen['Margen %'] < 0].copy()
                        if not productos_perdida.empty:
                            # Usar los nombres de columnas originales para el cálculo
                            coste_col_name = f'{coste_col} (€)'
                            precio_venta_col = 'Precio Venta (€)'
                            perdida_total = ((productos_perdida[coste_col_name] - productos_perdida[precio_venta_col]) * 
                                           abs(productos_perdida['Margen %'] / 100)).sum()
                            if pd.isna(perdida_total) or perdida_total == float('inf') or perdida_total == float('-inf'):
                                st.metric("Pérdida estimada", "N/A")
                            else:
                                st.metric("Pérdida estimada", f"{perdida_total:.0f}€")
                        else:
                            st.metric("Pérdida estimada", "0€")
                    except Exception as e:
                        st.metric("Pérdida estimada", f"Error: {str(e)}")
            else:
                st.info(f"No hay productos con margen inferior al {umbral_margen*100:.0f}%")
        else:
            st.info("No hay datos de Precio Coste, Beneficio o Cantidad disponibles para el análisis de márgenes.")

    elif seccion == "Análisis de Descripciones":
        # --- Análisis de Descripciones por Familia ---
        viz_title("Análisis de Descripciones por Familia")

        # Import preprocessing function
        try:
            from preprocess_descriptions import get_processed_descriptions
        except ImportError:
            st.error("❌ Preprocessing module not found. Please ensure 'preprocess_descriptions.py' is in the same directory.")
            return

        # Streamlit uploader for description files (multiple files)
        desc_files = st.file_uploader(
            "Sube el archivo de descripciones (Excel) - Puedes subir múltiples archivos",
            type=["xlsx"],
            accept_multiple_files=True,
            key="desc_file_uploader"
        )
        
        if desc_files:
            with st.spinner("🔄 Procesando archivos de descripciones..."):
                # Preprocess uploaded files
                df_desc = get_processed_descriptions(desc_files)
                
                if df_desc is not None and not df_desc.empty:
                    # NUEVA LISTA DE COLUMNAS DE DESCRIPCIÓN
                    desc_cols = ['MANGA', 'CUELLO', 'TEJIDO', 'DETALLE', 'ESTILO', 'CORTE']
                    
                    # Verificar que las columnas necesarias existen
                    required_cols = ['Código único', 'fashion_main_description_1'] + desc_cols
                    available_cols = [col for col in required_cols if col in df_desc.columns]
                    
                    if len(available_cols) >= 2:  # At least Código único and fashion_main_description_1
                        # --- FILTROS ---
                        # Verificar si hay una familia específica seleccionada en el filtro global
                        # Si df_ventas está filtrado por una familia específica, mostrar las descripciones
                        familias_unicas = df_ventas['Familia'].unique()
                        
                        if len(familias_unicas) == 1 and familias_unicas[0] != "Todas las familias":
                            # Hay una familia específica seleccionada
                            familia_actual = familias_unicas[0]
                            
                            # Solo mostrar el selector de tipo de descripción
                            opciones_desc = ["Descripción Completa"] + [col for col in desc_cols if col in df_desc.columns]
                            tipo_descripcion = st.selectbox(
                                "Selecciona Tipo de Descripción:", 
                                opciones_desc, 
                                key="tipo_desc_selector"
                            )

                            # --- GENERACIÓN DE DESCRIPCIONES ---
                            if tipo_descripcion == "Descripción Completa":
                                df_desc['Descripción Analizada'] = df_desc['fashion_main_description_1'].fillna('N/A')
                            else:
                                df_desc['Descripción Analizada'] = df_desc[tipo_descripcion].fillna('N/A')
                            
                            df_desc_clean = df_desc[['Código único', 'Descripción Analizada']].copy().dropna()

                            # Antes del merge de descripciones:
                            if 'Código único' in df_ventas.columns and 'Código único' in df_desc_clean.columns:
                                df_ventas['Codigo_merge'] = df_ventas['Código único'].astype(str).str[:13]
                                df_desc_clean['Código único'] = df_desc_clean['Código único'].astype(str).str.split().str[0]
                                ventas_con_desc = df_ventas.merge(
                                    df_desc_clean,
                                    left_on='Codigo_merge',
                                    right_on='Código único',
                                    how='left',
                                    suffixes=('', '_desc')
                                )
                                df_ventas.drop(columns=['Codigo_merge'], inplace=True)
                        
                            # FILTRO POR FAMILIA (usando el filtro global)
                            df_familia_desc = ventas_con_desc[ventas_con_desc['Familia'] == familia_actual]
                            
                            desc_group = df_familia_desc.groupby('Descripción Analizada').agg({
                                'Beneficio': 'sum',
                                'Cantidad': 'sum'
                            }).reset_index()
                            
                            # FILTRO DE DESCRIPCIONES VACÍAS
                            desc_group = desc_group[desc_group['Descripción Analizada'] != 'N/A']
                            desc_group = desc_group[desc_group['Descripción Analizada'].str.strip() != '']

                            if not desc_group.empty:
                                desc_group = desc_group.sort_values('Beneficio', ascending=False)
                                top10 = desc_group.head(10)
                                bottom10 = desc_group.tail(10)

                                # --- Side-by-side Top/Bottom 10 Descriptions ---
                                # --- GRÁFICOS DE TOP Y BOTTOM UNO DEBAJO DEL OTRO CON ALTURA DINÁMICA ---

                                # Configuración para altura dinámica
                                altura_por_fila = 40   # Altura estimada por barra
                                altura_minima = 400    # Altura mínima para que no quede muy apretado
                                altura_maxima = 800    # Altura máxima para que no sea exagerado

                                # Calculamos altura según la cantidad de barras
                                altura_top = min(max(len(top10) * altura_por_fila, altura_minima), altura_maxima)
                                altura_bottom = min(max(len(bottom10) * altura_por_fila, altura_minima), altura_maxima)

                                # --- GRÁFICO TOP 10 ---
                                viz_title(f'Top 10 en {tipo_descripcion} - {familia_actual}')
                                fig_top = px.bar(
                                    top10, 
                                    x='Beneficio', 
                                    y='Descripción Analizada', 
                                    orientation='h', 
                                    color='Beneficio', 
                                    color_continuous_scale=COLOR_GRADIENT,
                                    text='Cantidad'
                                )
                                fig_top.update_layout(
                                    showlegend=False, 
                                    height=altura_top,
                                    yaxis={'categoryorder':'total ascending', 'title': ''},
                                    margin=dict(t=30, b=0, l=0, r=0),
                                    paper_bgcolor="rgba(0,0,0,0)", 
                                    plot_bgcolor="rgba(0,0,0,0)"
                                )
                                fig_top.update_traces(
                                    texttemplate='%{text:,.0f} uds', 
                                    textposition='outside', 
                                    hovertemplate="Descripción: %{y}<br>Ventas: %{x:,.2f}€<br>Unidades: %{text:,.0f}<extra></extra>",
                                    opacity=0.8
                                )
                                st.plotly_chart(fig_top, use_container_width=True, key=f"top10_{tipo_descripcion}_{familia_actual}")

                                # --- GRÁFICO BOTTOM 10 ---
                                viz_title(f'Bottom 10 en {tipo_descripcion} - {familia_actual}')
                                fig_bottom = px.bar(
                                    bottom10, 
                                    x='Beneficio', 
                                    y='Descripción Analizada', 
                                    orientation='h', 
                                    color='Beneficio', 
                                    color_continuous_scale=COLOR_GRADIENT,
                                    text='Cantidad'
                                )
                                fig_bottom.update_layout(
                                    showlegend=False, 
                                    height=altura_bottom,
                                    yaxis={'categoryorder':'total ascending', 'title': ''},
                                    margin=dict(t=30, b=0, l=0, r=0),
                                    paper_bgcolor="rgba(0,0,0,0)", 
                                    plot_bgcolor="rgba(0,0,0,0)"
                                )
                                fig_bottom.update_traces(
                                    texttemplate='%{text:,.0f} uds', 
                                    textposition='outside', 
                                    hovertemplate="Descripción: %{y}<br>Ventas: %{x:,.2f}€<br>Unidades: %{text:,.0f}<extra></extra>",
                                    opacity=0.8
                                )
                                st.plotly_chart(fig_bottom, use_container_width=True, key=f"bottom10_{tipo_descripcion}_{familia_actual}")
                            else:
                                st.info(f"No hay datos de '{tipo_descripcion}' para la familia '{familia_actual}'.")
                        else:
                            # No hay familia específica seleccionada
                            st.info("Selecciona una familia en el filtro general para poder ver las descripciones")
                    else:
                        st.warning(f"Columnas requeridas no encontradas. Se necesitan al menos: Código único y fashion_main_description_1")
                else:
                    st.error("❌ Error en el procesamiento de los archivos de descripciones.")
        else:
            st.warning("Por favor, sube uno o más archivos de descripciones (.xlsx) para ver el análisis de descripciones.")

# Cached function for calculating store rankings
@st.cache_data
def calculate_store_rankings(df_ventas):
    """Cache the store ranking calculations"""
    ventas_por_tienda = df_ventas.groupby('Tienda').agg({
        'Cantidad': 'sum',
        'Beneficio': 'sum'
    }).reset_index()
    ventas_por_tienda.columns = ['Tienda', 'Unidades Vendidas', 'Beneficio']
    
    # Ordenar por Beneficio para obtener el ranking
    ventas_por_tienda = ventas_por_tienda.sort_values('Beneficio', ascending=False).reset_index(drop=True)
    ventas_por_tienda['Ranking'] = ventas_por_tienda.index + 1
    return ventas_por_tienda

# Cached function for calculating family rankings per store
@st.cache_data
def calculate_family_rankings(df_ventas):
    """Cache the family ranking calculations per store"""
    familias_por_tienda = df_ventas.groupby(['Tienda', 'Familia'])['Cantidad'].sum().reset_index()
    familias_por_tienda = familias_por_tienda.sort_values('Cantidad', ascending=False)
    return familias_por_tienda

@st.cache_data
def preprocess_ventas_data(df_ventas):
    """Cache the data preprocessing to avoid reprocessing on every interaction - OPTIMIZED VERSION"""
    if df_ventas.empty:
        return df_ventas
    
    df_ventas = df_ventas.copy()
    column_map = {
    "TPV": "Código Tienda",
    "NombreTPV": "Tienda",
    "Zona geográfica": "Zona geográfica",
    "Fecha Documento": "Fecha venta",
    "Marca": "Código Marca",
    "Descripción Marca": "Marca",
    "Temporada": "Temporada",
    "Genérico": "Genérico",
    "ACT": "Código único",
    "Artículo": "Artículo",
    "Modelo Artículo": "Modelo Artículo",
    "Color": "Código Color",
    "Descripción Color": "Color",
    "Talla": "Talla",
    "Familia": "Código Familia",
    "Descripción Familia": "Familia",
    "Tema": "Tema",
    "Cantidad": "Cantidad",
    "P.V.P.": "PVP",
    "Subtotal": "Beneficio"
    }

    # OPTIMIZATION: Only rename columns that exist
    existing_columns = {k: v for k, v in column_map.items() if k in df_ventas.columns}
    df_ventas = df_ventas.rename(columns=existing_columns)
    
    # OPTIMIZATION: Process date column more efficiently
    if 'Fecha venta' in df_ventas.columns:
        df_ventas['Fecha venta'] = pd.to_datetime(df_ventas['Fecha venta'], format='%d/%m/%Y', errors='coerce')
        df_ventas = df_ventas.dropna(subset=['Fecha venta'])
        df_ventas['Mes'] = df_ventas['Fecha venta'].dt.to_period('M').astype(str)

    # OPTIMIZATION: Process code columns more efficiently
    if 'Código único' in df_ventas.columns:
        df_ventas['Código único'] = df_ventas['Código único'].astype(str).str.split().str[0]

    
    if 'Familia' in df_ventas.columns:
        df_ventas['Familia'] = df_ventas['Familia'].fillna("Sin Familia")
    
    # OPTIMIZATION: Process numeric columns more efficiently
    numeric_columns = ['Cantidad', 'Beneficio', 'PVP']
    for col in numeric_columns:
        if col in df_ventas.columns:
            df_ventas[col] = pd.to_numeric(df_ventas[col], errors='coerce').fillna(0)
    
    # OPTIMIZATION: Handle color column more efficiently
    if 'Color' not in df_ventas.columns:
        df_ventas['Color'] = 'Desconocido'
    
    # OPTIMIZATION: Handle season column more efficiently
    if 'Temporada' not in df_ventas.columns:
        temporada_columns = [col for col in df_ventas.columns if 'temporada' in col.lower() or 'season' in col.lower()]
        if temporada_columns:
            df_ventas['Temporada'] = df_ventas[temporada_columns[0]]
        else:
            df_ventas['Temporada'] = 'Sin Temporada'
    else:
        df_ventas['Temporada'] = df_ventas['Temporada'].fillna('Sin Temporada')
    
    # OPTIMIZATION: Identify online stores more efficiently
    if 'Tienda' in df_ventas.columns:
        df_ventas['Es_Online'] = df_ventas['Tienda'].str.contains('ONLINE', case=False, na=False)
    else:
        df_ventas['Es_Online'] = False
    
    return df_ventas

# Cached function for data preprocessing
@st.cache_data
def preprocess_productos_data(df_productos):
    """Cache the data preprocessing to avoid reprocessing on every interaction - OPTIMIZED VERSION"""
    if df_productos.empty:
        return df_productos
    
    df_productos = df_productos.copy()
    column_map_productos = {
        "TPV": "Código Tienda",
        "NombreTPV": "Tienda",
        "Fecha Presupuesto": "Fecha Presupuesto",
        "Fecha Tope": "Fecha Tope",
        "Marca": "Código Marca",
        "Descripción Marca": "Marca",
        "Generico": "Genérico",
        "ACT": "Código único",
        "Artículo": "Artículo",
        "Modelo Artículo": "Modelo Artículo",
        "Color": "Código Color",
        "Descripción Color": "Color",
        "Talla": "Talla",
        "Tema": "Tema",
        "Unnamed: 14": "Unnamed: 14",
        "Cantidad Pedida": "Cantidad pedida",
        "Fecha REAL entrada en almacén": "Fecha almacén",
        "Precio Coste": "Precio Coste",
        "P.V.P.": "PVP",
        "Importe de Coste": "Importe de Coste"
    }
    
    # OPTIMIZATION: Only rename columns that exist
    existing_columns = {k: v for k, v in column_map_productos.items() if k in df_productos.columns}
    df_productos = df_productos.rename(columns=existing_columns)
    
    # OPTIMIZATION: Process date column more efficiently
    if 'Fecha almacén' in df_productos.columns:
        df_productos['Fecha almacén'] = pd.to_datetime(df_productos['Fecha almacén'], format='%d/%m/%Y', errors='coerce')
        df_productos = df_productos.dropna(subset=['Fecha almacén'])
        df_productos['Mes'] = df_productos['Fecha almacén'].dt.to_period('M').astype(str)

    # OPTIMIZATION: Process code columns more efficiently
    if 'Código único' in df_productos.columns:
        df_productos['Código único'] = df_productos['Código único'].astype(str).str.split().str[0]

    
    # OPTIMIZATION: Process numeric columns more efficiently
    numeric_columns = ['Cantidad pedida', 'PVP']
    for col in numeric_columns:
        if col in df_productos.columns:
            df_productos[col] = pd.to_numeric(df_productos[col], errors='coerce').fillna(0)
    
    # OPTIMIZATION: Process theme column more efficiently
    if 'Tema' in df_productos.columns:
        df_productos['Tema_temporada'] = df_productos['Tema'].astype(str).str[:6]
    
    # OPTIMIZATION: Handle color column more efficiently
    if 'Color' not in df_productos.columns:
        df_productos['Color'] = 'Desconocido'
    
    return df_productos

@st.cache_data
def preprocess_traspasos_data(df_traspasos):
    """Cache the data preprocessing to avoid reprocessing on every interaction - OPTIMIZED VERSION"""
    if df_traspasos.empty:
        return df_traspasos
    
    df_traspasos = df_traspasos.copy()
    column_map_traspasos = {
        "Nº. TPV Origen": "Nº. TPV Origen",
        "NombreTPVOrigen": "NombreTPVOrigen",
        "Fecha Documento": "Fecha enviado",
        "Nº. TPV Destino": "Nº. TPV Destino",
        "NombreTpvDestino": "Tienda",
        "Zona geográfica": "Zona geográfica",
        "Marca": "Marca",
        "Descripción Marca": "Descripción Marca",
        "Temporada": "Temporada",
        "Genérico": "Genérico",
        "ACT": "Código único",
        "Artículo": "Artículo",
        "Modelo Artículo": "Modelo Artículo",
        "Color": "Código Color",
        "Descripción Color": "Descripción Color",
        "Talla": "Talla",
        "Enviado": "Cantidad enviada",
        "Descripción Familia": "Familia"
    }
    
    # OPTIMIZATION: Only rename columns that exist
    existing_columns = {k: v for k, v in column_map_traspasos.items() if k in df_traspasos.columns}
    df_traspasos = df_traspasos.rename(columns=existing_columns)
  
    # OPTIMIZATION: Process date column more efficiently
    if 'Fecha enviado' in df_traspasos.columns:
        df_traspasos['Fecha enviado'] = pd.to_datetime(df_traspasos['Fecha enviado'], format='%d/%m/%Y', errors='coerce')
        df_traspasos = df_traspasos.dropna(subset=['Fecha enviado'])
        df_traspasos['Mes'] = df_traspasos['Fecha enviado'].dt.to_period('M').astype(str)

    # OPTIMIZATION: Process code columns more efficiently
    if 'Código único' in df_traspasos.columns:
        df_traspasos['Código único'] = df_traspasos['Código único'].astype(str).str.split().str[0]
    
    # OPTIMIZATION: Process numeric columns more efficiently
    if 'Cantidad enviada' in df_traspasos.columns:
        df_traspasos['Cantidad enviada'] = pd.to_numeric(df_traspasos['Cantidad enviada'], errors='coerce').fillna(0)
    
    # OPTIMIZATION: Handle color column more efficiently
    if 'Color' not in df_traspasos.columns:
        df_traspasos['Color'] = 'Desconocido'
    
    return df_traspasos

# Cached function for consistent temporada colors
@st.cache_data
def get_temporada_colors(df_ventas):
    """Get consistent color mapping for temporadas across all charts"""
    temporadas = sorted(df_ventas['Temporada'].unique())
    color_mapping = {}
    for i, temp in enumerate(temporadas):
        color_mapping[temp] = TEMPORADA_COLORS[i % len(TEMPORADA_COLORS)]
    return color_mapping

# New cached functions for Resumen General optimization
@st.cache_data
def calculate_rotation_metrics(df_productos, df_traspasos, df_ventas):
    """Cache the rotation calculation which is very expensive - OPTIMIZED VERSION"""
    if df_productos.empty or 'Fecha almacén' not in df_productos.columns:
        return None, None, None, None, None, None, None, None, None, None, None, None
    
    # Prepare data for rotation calculation - OPTIMIZED
    df_productos_rotacion = df_productos[['Código único', 'Talla', 'Fecha almacén']].copy()
    df_productos_rotacion['Fecha almacén'] = pd.to_datetime(df_productos_rotacion['Fecha almacén'], format='%d/%m/%Y', errors='coerce')
    
    df_traspasos_rotacion = df_traspasos[['Código único', 'Talla', 'Tienda', 'Fecha enviado']].copy()
    df_traspasos_rotacion['Fecha enviado'] = pd.to_datetime(df_traspasos_rotacion['Fecha enviado'], format='%d/%m/%Y', errors='coerce')
    
    ventas_rotacion = df_ventas[['Código único', 'Talla', 'Tienda', 'Fecha venta', 'Familia']].copy()
    ventas_rotacion['Fecha venta'] = pd.to_datetime(ventas_rotacion['Fecha venta'], format='%d/%m/%Y', errors='coerce')
    
    # Filter out invalid dates early for better performance
    df_productos_rotacion = df_productos_rotacion.dropna(subset=['Fecha almacén'])
    df_traspasos_rotacion = df_traspasos_rotacion.dropna(subset=['Fecha enviado'])
    ventas_rotacion = ventas_rotacion.dropna(subset=['Fecha venta'])
    
    if df_productos_rotacion.empty or df_traspasos_rotacion.empty or ventas_rotacion.empty:
        return None, None, None, None, None, None, None, None, None, None, None, None
    
    # OPTIMIZED: Use only necessary columns for merge
    ventas_con_entrada = ventas_rotacion.merge(
        df_productos_rotacion,
        on=['Código único'],
        how='inner'
    )
    
    if ventas_con_entrada.empty:
        return None, None, None, None, None, None, None, None, None, None, None, None
    
    # OPTIMIZED: Merge with traspasos using only necessary columns
    rotacion_completa = ventas_con_entrada.merge(
        df_traspasos_rotacion,
        on=['Código único', 'Tienda'],
        how='inner'
    )
    
    if rotacion_completa.empty:
        return None, None, None, None, None, None, None, None, None, None, None, None
    
    # Calculate rotation days with validation
    rotacion_completa['Dias_Rotacion'] = (
        rotacion_completa['Fecha venta'] - rotacion_completa['Fecha almacén']
    ).dt.days
    
    # Filter valid rotation days (0-365 days to avoid extreme outliers)
    rotacion_completa = rotacion_completa[
        (rotacion_completa['Dias_Rotacion'] >= 0) & 
        (rotacion_completa['Dias_Rotacion'] <= 365)
    ]
    
    if rotacion_completa.empty:
        return None, None, None, None, None, None, None, None, None, None, None, None
    
    # Calculate comprehensive rotation metrics by store
    rotacion_por_tienda = rotacion_completa.groupby('Tienda').agg({
        'Dias_Rotacion': ['mean', 'median', 'std', 'count']
    }).reset_index()
    rotacion_por_tienda.columns = ['Tienda', 'Dias_Promedio', 'Dias_Mediana', 'Dias_Std', 'Productos_Con_Rotacion']
    
    # Calculate comprehensive rotation metrics by product
    rotacion_por_producto = rotacion_completa.groupby(['Código único', 'Familia']).agg({
        'Dias_Rotacion': ['mean', 'median', 'std', 'count']
    }).reset_index()
    rotacion_por_producto.columns = ['Código único', 'Familia', 'Dias_Promedio', 'Dias_Mediana', 'Dias_Std', 'Ventas_Con_Rotacion']
    
    # Calculate overall statistics
    dias_rotacion_global = rotacion_completa['Dias_Rotacion']
    promedio_global = dias_rotacion_global.mean()
    mediana_global = dias_rotacion_global.median()
    std_global = dias_rotacion_global.std()
    
    # Calculate KPIs with better logic
    tienda_mayor_rotacion = "Sin datos"
    tienda_mayor_rotacion_dias = 0
    tienda_menor_rotacion = "Sin datos"
    tienda_menor_rotacion_dias = 0
    producto_mayor_rotacion = "Sin datos"
    producto_mayor_rotacion_dias = 0
    producto_menor_rotacion = "Sin datos"
    producto_menor_rotacion_dias = 0
    
    if not rotacion_por_tienda.empty:
        # Filter stores with minimum data points for reliability
        tiendas_confiables = rotacion_por_tienda[rotacion_por_tienda['Productos_Con_Rotacion'] >= 5]
        
        if not tiendas_confiables.empty:
            # Store with highest rotation (lowest median days - more reliable than mean)
            idx_mayor = tiendas_confiables['Dias_Mediana'].idxmin()
            tienda_mayor_rotacion = tiendas_confiables.loc[idx_mayor, 'Tienda']
            tienda_mayor_rotacion_dias = tiendas_confiables.loc[idx_mayor, 'Dias_Mediana']
            
            # Store with lowest rotation (highest median days)
            idx_menor = tiendas_confiables['Dias_Mediana'].idxmax()
            tienda_menor_rotacion = tiendas_confiables.loc[idx_menor, 'Tienda']
            tienda_menor_rotacion_dias = tiendas_confiables.loc[idx_menor, 'Dias_Mediana']
    
    if not rotacion_por_producto.empty:
        # Filter products with minimum data points for reliability
        productos_confiables = rotacion_por_producto[rotacion_por_producto['Ventas_Con_Rotacion'] >= 3]
        
        if not productos_confiables.empty:
            # Product with highest rotation (lowest median days)
            idx_mayor = productos_confiables['Dias_Mediana'].idxmin()
            producto_mayor_rotacion = productos_confiables.loc[idx_mayor, 'Familia']
            producto_mayor_rotacion_dias = productos_confiables.loc[idx_mayor, 'Dias_Mediana']
            
            # Product with lowest rotation (highest median days)
            idx_menor = productos_confiables['Dias_Mediana'].idxmax()
            producto_menor_rotacion = productos_confiables.loc[idx_menor, 'Familia']
            producto_menor_rotacion_dias = productos_confiables.loc[idx_menor, 'Dias_Mediana']
    
    return (
        tienda_mayor_rotacion, tienda_mayor_rotacion_dias, 
        tienda_menor_rotacion, tienda_menor_rotacion_dias,
        producto_mayor_rotacion, producto_mayor_rotacion_dias, 
        producto_menor_rotacion, producto_menor_rotacion_dias,
        promedio_global, mediana_global, std_global, len(rotacion_completa)
    )

@st.cache_data
def calculate_basic_kpis(df_ventas):
    """Cache basic KPI calculations"""
    total_ventas_dinero = df_ventas['Beneficio'].sum()
    total_familias = df_ventas['Familia'].nunique()
    
    # Calculate returns (monetary amount of negative quantities)
    devoluciones = df_ventas[df_ventas['Cantidad'] < 0]
    total_devoluciones_dinero = abs(devoluciones['Beneficio'].sum())
    
    # Separate physical and online stores
    ventas_fisicas = df_ventas[~df_ventas['Es_Online']]
    ventas_online = df_ventas[df_ventas['Es_Online']]
    
    # Calculate KPIs by store type
    ventas_fisicas_dinero = ventas_fisicas['Beneficio'].sum()
    ventas_online_dinero = ventas_online['Beneficio'].sum()
    tiendas_fisicas = ventas_fisicas['Tienda'].nunique()
    tiendas_online = ventas_online['Tienda'].nunique()
    
    return (total_ventas_dinero, total_devoluciones_dinero, total_familias, 
            ventas_fisicas_dinero, ventas_online_dinero, tiendas_fisicas, tiendas_online)

@st.cache_data
def calculate_monthly_sales_data(df_ventas):
    """Cache monthly sales data calculation"""
    ventas_mes_tipo = df_ventas.groupby(['Mes', 'Es_Online']).agg({
        'Cantidad': 'sum',
        'Beneficio': 'sum'
    }).reset_index()
    
    
    ventas_mes_tipo['Tipo'] = ventas_mes_tipo['Es_Online'].map({True: 'Online', False: 'Física'})
    
    return ventas_mes_tipo
