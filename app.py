# Prueba de actualización
import streamlit as st
st.set_page_config(page_title="TRUCCO", page_icon="🡕", layout="wide")

from dashboard import mostrar_dashboard
import pandas as pd
import base64
import os

# Performance optimization: Set pandas options
pd.options.mode.chained_assignment = None  # default='warn'

# Function to get absolute path for assets
def get_asset_path(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "assets", filename)

# Cached function for loading Excel data
@st.cache_data
def load_excel_data(file):
    """Cache the Excel file loading to avoid reprocessing on every interaction - OPTIMIZED VERSION"""
    try:
        # OPTIMIZATION: Use more efficient Excel reading
        xls = pd.ExcelFile(file, engine="openpyxl")
        
        # OPTIMIZATION: Read only necessary sheets and optimize data types
        df_productos = pd.read_excel(
            xls, 
            sheet_name="Compra",
            dtype={
                'ACT': str,
                'Cantidad Pedida': 'Int64',
                'P.V.P.': 'Float64',
                'url_image': str  # Add url_image column
            },
            na_values=['', 'nan', 'NaN'],
            keep_default_na=False
        )
        
        df_traspasos = pd.read_excel(
            xls, 
            sheet_name="Traspasos de almacén a tienda",
            dtype={
                'ACT': str,
                'Enviado': 'Int64',
                'url_image': str  # Add url_image column
            },
            na_values=['', 'nan', 'NaN'],
            keep_default_na=False
        )
        
        df_ventas = pd.read_excel(
            xls, 
            sheet_name="ventas 23 24 25",
            dtype={
                'ACT': str,
                'Cantidad': 'Int64',
                'P.V.P.': 'Float64',
                'Subtotal': 'Float64',
                'url_image': str,  # Add url_image column
                'url_thumbnail': str  # Add url_thumbnail column
            },
            na_values=['', 'nan', 'NaN'],
            keep_default_na=False
        )
        
        # OPTIMIZATION: Early data cleaning and type conversion
        for df, df_name in [(df_productos, 'Productos'), (df_traspasos, 'Traspasos'), (df_ventas, 'Ventas')]:
            # Convert string columns to more efficient types
            for col in df.columns:
                if df[col].dtype == 'object' and col != 'url_image':  # Preserve url_image as string
                    # Check if column contains mostly numeric data
                    numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
                    if numeric_count > len(df) * 0.8:  # If 80%+ is numeric
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    else:
                        # Convert to string and handle NaN values
                        df[col] = df[col].astype(str).replace('nan', '')
            
            # OPTIMIZATION: Remove completely empty rows early
            df.dropna(how='all', inplace=True)
            
            # OPTIMIZATION: Remove completely empty columns early (except url_image)
            empty_cols = df.columns[df.isna().all()]
            empty_cols = [col for col in empty_cols if col != 'url_image']
            df.drop(columns=empty_cols, inplace=True)
        
        return df_productos, df_traspasos, df_ventas
        
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        # Return empty DataFrames with proper structure
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Cached function for filtering data by season
@st.cache_data
def filter_by_season(df_ventas, temporada_seleccionada):
    """Cache the season filtering to avoid reprocessing"""
    if temporada_seleccionada != "Todas las temporadas":
        return df_ventas[df_ventas["Temporada"] == temporada_seleccionada]
    return df_ventas

# Cached function for filtering data by family
@st.cache_data
def filter_by_family(df_ventas, familia_seleccionada):
    """Cache the family filtering to avoid reprocessing"""
    if familia_seleccionada != "Todas las familias":
        return df_ventas[df_ventas["Descripción Familia"] == familia_seleccionada]
    return df_ventas

# Estilos CSS
st.markdown("""
    <style>
    .header-container {
        display: flex;
        align-items: center;
        padding: 20px 0;
        margin-bottom: 40px;
    }
    .logo-container {
        margin-right: 30px;
    }
    .main-title {
        font-size: 48px;
        color: #666666;
        font-weight: 600;
        line-height: 1;
        letter-spacing: -1px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin: 0;
        padding: 0;
    }
    .login-title {
        font-size: 26px;
        font-weight: 500;
        margin-bottom: 1rem;
        color: white;
    }
    .stApp {
        background-color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Función para aplicar fondo
def set_background(image_file):
    """Establece una imagen de fondo para la app y aplica estilos de login."""
    try:
        with open(image_file, "rb") as f:
            img_data = f.read()
            b64_encoded = base64.b64encode(img_data).decode()
            style = f"""
                <style>
                .stApp {{
                    background-image: url(data:image/png;base64,{b64_encoded});
                    background-size: cover;
                }}
                .login-container {{
                    background-color: rgba(255, 255, 255, 0.8);
                    padding: 30px;
                    border-radius: 10px;
                    text-align: center;
                }}
                .login-title {{
                    font-size: 24px;
                    font-weight: bold;
                    color: white;
                    margin-bottom: 20px;
                }}
                /* Poner etiquetas de input en blanco */
                div[data-testid="stTextInput"] label {{
                    color: white;
                }}
                </style>
            """
            st.markdown(style, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading background image: {e}")

# Login de seguridad
if 'logueado' not in st.session_state:
    try:
        st.markdown(f'''
            <div style="display: flex; align-items: center; width: 100%; margin-bottom: 40px; margin-top: 0; padding-top: 0;">
                <img src="data:image/png;base64,{base64.b64encode(open(get_asset_path('Logo.png'), 'rb').read()).decode()}" style="height: 160px; margin-right: 48px; margin-top: 0; padding-top: 0;" />
                <img src="data:image/png;base64,{base64.b64encode(open(get_asset_path('fondo.png'), 'rb').read()).decode()}" style="height: 240px; width: 100%; object-fit: cover; margin-top: 0; padding-top: 0;" />
            </div>
            <div style='text-align: left; color: white; font-size: 22px; font-weight: 400; margin-left: 8px; margin-bottom: 16px;'>Acceso a Trucco Analytics</div>
        ''', unsafe_allow_html=True)
        usuario = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        if st.button("Entrar"):
            if usuario and password:
                st.session_state['logueado'] = True
                st.rerun()
            else:
                st.warning("Por favor, introduce usuario y contraseña")
    except Exception as e:
        st.error(f"Error loading login assets: {e}")

else:
    # Ya logueado
    try:
        with open(get_asset_path("Logo.png"), "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
            st.markdown(f"""
                <div class="header-container">
                    <div class="logo-container">
                        <img src="data:image/png;base64,{logo_data}" width="100">
                    </div>
                    <h1 class="main-title" style='font-size:32px;'>Plataforma de Análisis y Predicción</h1>
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading header logo: {e}")

    st.sidebar.title("Menú de Navegación")
    
    # Add session reset button
    if st.sidebar.button("🔄 Resetear sesión"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()
    
    opcion = st.sidebar.radio("Selecciona una vista", ["Análisis", "Predicción"])

    if opcion == "Análisis":
        # Subida de archivo solo para análisis
        file = st.sidebar.file_uploader("Sube el archivo Excel", type=["xlsx"])

        if file:
            try:
                # Use session state to avoid reloading data if file hasn't changed
                file_hash = hash(file.getvalue())
                if 'file_hash' not in st.session_state or st.session_state.file_hash != file_hash:
                    with st.spinner("Cargando y procesando datos..."):
                        st.session_state.file_hash = file_hash
                        st.session_state.df_productos, st.session_state.df_traspasos, st.session_state.df_ventas = load_excel_data(file)
                    st.sidebar.success("Archivo cargado correctamente")
                
                df_productos = st.session_state.df_productos
                df_traspasos = st.session_state.df_traspasos
                df_ventas = st.session_state.df_ventas

                seccion = st.sidebar.selectbox("Área de Análisis", [
                    "Resumen General",
                    "Geográfico y Tiendas",
                    "Producto, Campaña, Devoluciones y Rentabilidad",
                    "Análisis con fotos"
                ])
                st.sidebar.header("Filtros")
                # --- Filtro de temporada ---
                temporadas = df_ventas["Temporada"].dropna().unique().tolist()
                temporadas.sort()
                temporadas_opciones = ["Todas las temporadas"] + temporadas
                temporada_seleccionada = st.sidebar.selectbox("Temporada", temporadas_opciones)
                if temporada_seleccionada != "Todas las temporadas":
                    df_ventas = filter_by_season(df_ventas, temporada_seleccionada)
                # --- Fin filtro de temporada ---
                
                # --- Filtro de familia ---
                familias = df_ventas["Descripción Familia"].dropna().unique().tolist()
                familias.sort()

                # Agregar las dos opciones especiales al principio
                familias_opciones = ["Todas las familias", "Todas sin GR.ART.FICTICIO"] + familias

                familia_seleccionada = st.sidebar.selectbox("Familia", familias_opciones)

                if familia_seleccionada == "Todas las familias":
                    # No filtrar nada
                    pass

                elif familia_seleccionada == "Todas sin GR.ART.FICTICIO":
                    # Filtrar excluyendo GR.ART.FICTICIO
                    df_ventas = df_ventas[df_ventas["Descripción Familia"] != "GR.ART.FICTICIO"]

                else:
                    # Filtrar solo la familia seleccionada
                    df_ventas = filter_by_family(df_ventas, familia_seleccionada)
                # --- Fin filtro de familia ---

                with st.spinner("Generando dashboard..."):
                    mostrar_dashboard(df_productos, df_traspasos, df_ventas, seccion)

            except Exception as e:
                st.error(f"Error al procesar el archivo: {e}")
        else:
            st.info("Sube el archivo Excel para comenzar el análisis.")
    
   
