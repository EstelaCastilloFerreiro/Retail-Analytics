import pandas as pd
import re
import spacy
from spacy.pipeline import EntityRuler
from spacy.matcher import PhraseMatcher
import streamlit as st

def preprocess_description_files(uploaded_files):
    """
    Preprocess uploaded description files and return processed DataFrame
    
    Args:
        uploaded_files: List of uploaded file objects from Streamlit
        
    Returns:
        processed_df: DataFrame with extracted features ready for dashboard
    """
    
    if not uploaded_files:
        return None
    
    # Step 1: Load and combine all uploaded files
    dfs = []
    for file in uploaded_files:
        try:
            df = pd.read_excel(file, engine='openpyxl')
            dfs.append(df)
            st.success(f"✅ Loaded: {file.name}")
        except Exception as e:
            st.error(f"❌ Error reading {file.name}: {e}")
    
    if not dfs:
        st.error("No files were successfully loaded.")
        return None
    
    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    st.success(f"✅ Combined DataFrame shape: {combined_df.shape}")
    
    # Step 2: Check required columns
    required_columns = [
        'provider_ref', 'fashion_main_description_1', 'title', 'fashion_season', 
        'fashion_season_year', 'precio_pvp', 'fashion_compo_material_1', 
        'fashion_compo_percentage_1', 'fashion_compo_unit_1'
    ]
    
    # Filter to only required columns if they exist
    existing_columns = [col for col in required_columns if col in combined_df.columns]
    if not existing_columns:
        st.error("No required columns found in uploaded files.")
        return None
    
    df_filtered = combined_df[existing_columns]

    # Step 3: Define dictionaries and functions
    diccionario = {
        # Cuellos
        "cuello v": "cuello en v",
        "cuello en v": "cuello en v",
        "cuello pico": "cuello pico",
        "cuello redondo": "cuello redondo",
        "cuello redondo cerrado": "cuello redondo cerrado",
        "cuello solapa": "cuello solapa",
        "cuello halter": "cuello halter",
        "cuello mao": "cuello mao",
        "cuello vuelto": "cuello vuelto",
        "cuello perkins": "cuello perkins",
        "cuello barco": "cuello barco",
        "cuello cuadrado": "cuello cuadrado",
        "cuello asimétrico": "cuello asimétrico",
        "cuello alto": "cuello alto",
        "cuello cisne": "cuello cisne",
        "cuello tortuga": "cuello cisne",
        "cuello drapeado": "cuello drapeado",
        "cuello chimenea": "cuello chimenea",
        "cuello smoking": "cuello smoking",
        "cuello camisero": "cuello camisero",
        "cuello polo": "cuello polo",
        "cuello con volante": "cuello con volante",
        "cuello con lazo": "cuello con lazo",
        "cuello corazón": "cuello corazón",
        "cuello sweetheart": "cuello corazón",
        "cuello off shoulder": "cuello off shoulder",
        "cuello palabra de honor": "cuello palabra de honor",
        "cuello cruzado": "cuello cruzado",
        "cuello ilusión": "cuello ilusión",

        # Mangas
        "manga corta": "manga corta",
        "manga larga": "manga larga",
        "manga francesa": "manga francesa",
        "manga tres cuartos": "manga tres cuartos",
        "sin mangas": "sin mangas",
        "manga abombada": "manga abombada",
        "manga abullonada": "manga abullonada",
        "manga con volante": "manga con volante",
        "manga de volantes": "manga con volante",
        "manga corta avolantada": "manga con volante",
        "manga con puño": "manga con puño",
        "manga murciélago": "manga murciélago",
        "manga acampanada": "manga acampanada",
        "manga amplia": "manga amplia",
        "manga farol": "manga farol",
        "manga globo": "manga globo",
        "manga jamón": "manga jamón",
        "manga kimono": "manga kimono",
        "manga ranglán": "manga ranglán",
        "manga plisada": "manga plisada",
        "manga con abertura": "manga con abertura",
        "manga caída": "manga caída",
        "manga de encaje": "manga de encaje",
        "manga sin costuras": "manga sin costuras",
        "manga francesa amplia": "manga francesa",
        "manga corta estilizada": "manga corta",
        "manga larga abombada": "manga larga abombada",
        "manga larga abullonada": "manga larga abullonada",
        "manga larga con puño fruncido": "manga larga con puño fruncido",

        # Tejidos y Materiales
        "punto": "punto",
        "jacquard": "jacquard",
        "tejido jacquard": "jacquard",
        "canalé": "canalé",
        "tejido canalé": "canalé",
        "satinado": "satinado",
        "tejido satinado": "satinado",
        "algodón": "algodón",
        "poliéster": "poliéster",
        "terciopelo": "terciopelo",
        "denim": "denim",
        "lino": "lino",
        "seda": "seda",
        "crepé": "crepé",
        "bordado suizo": "bordado suizo",
        "tul": "tul",
        "encaje": "encaje",
        "tejido fluido": "tejido fluido",
        "tejido ligero": "tejido ligero",
        "tejido jaspeado": "tejido jaspeado",
        "tejido texturado": "tejido texturado",
        "tejido de fibra natural": "tejido de fibra natural",
        "tejido lurex": "tejido lurex",

        # Estampados
        "estampado abstracto": "estampado abstracto",
        "estampado floral": "estampado floral",
        "estampado geométrico": "estampado geométrico",
        "estampado vegetal": "estampado vegetal",
        "estampado botánico": "estampado vegetal",
        "estampado animal print": "estampado animal print",
        "estampado de rayas": "estampado de rayas",
        "estampado de hojas tropicales": "estampado de hojas tropicales",
        "estampado de hojas": "estampado de hojas",
        "estampado de palmeras": "estampado de palmeras",
        "estampado jaspeado": "estampado jaspeado",
        "estampado de círculos": "estampado de círculos",
        "estampado étnico": "estampado étnico",
        "estampado de abecedario": "estampado de abecedario",

        # Detalles y Acabados
        "lazo": "lazo",
        "lazo en bajo lateral": "lazo en bajo lateral",
        "lazo lateral": "lazo lateral",
        "lazo en cintura": "lazo en cintura",
        "volante": "volante",
        "fruncido": "fruncido",
        "fruncido en bajo": "fruncido en bajo",
        "fruncido en escote": "fruncido en escote",
        "fruncido elástico": "fruncido elástico",
        "plisado": "plisado",
        "drapeado": "drapeado",
        "abullonado": "abullonado",
        "asimétrico": "asimétrico",
        "acampanado": "acampanado",
        "con bolsillos": "con bolsillos",
        "sin bolsillos": "sin bolsillos",
        "bolsillos": "con bolsillos",
        "con botones": "con botones",
        "botones": "con botones",
        "sin botones": "sin botones",
        "con lazo": "con lazo",
        "con escote": "con escote",
        "con pinzas": "con pinzas",
        "con abertura": "con abertura",
        "sin costuras": "sin costuras",
        "con volantes": "con volantes",
        "con puños ajustados": "con puños ajustados",
        "sin tirantes": "sin tirantes",
        "tirantes": "tirantes",
        "tirantes anchos": "tirantes anchos",
        "strapless": "sin tirantes",
        "cut out": "cut out",
        "cut out en hombro": "cut out en hombro",
        "cut out en escote": "cut out en escote",
        "hombro descubierto": "hombro descubierto",
        "hombro asimétrico": "hombro asimétrico",
        "detalle de lentejuelas": "detalle de lentejuelas",
        "detalles de lentejuelas": "detalle de lentejuelas",
        "detalle de bordado": "detalle de bordado",
        "pelo": "pelo",
        "lentejuelas": "lentejuelas",
        "tachuelas": "tachuelas",

        # Cierres
        "cierre con botones": "con botones",
        "cierre con botón": "con botones",
        "cierre con gancho": "cierre con gancho",
        "cierre lateral en cinturilla": "cierre lateral en cinturilla",
        "cierre cruzado estilo caballero": "cierre cruzado estilo caballero",
        "cierre invisible": "cremallera invisible",
        "cierre con cremallera": "cremallera",
        "cierre con cremallera oculta": "cremallera invisible",
        "cierre oculto": "cremallera invisible",
        "cierre con lazo": "con lazo",
        "con cierre invisible": "cremallera invisible",

        # Siluetas y Cortes
        "cropped": "cropped",
        "crop": "cropped",
        "oversized": "oversized",
        "ajustado": "ajustado",
        "ceñido": "ajustado",
        "holgado": "holgado",
        "fluido": "fluido",
        "estructurado": "estructurado",
        "clásico": "clásico",
        "moderno": "moderno",
        "recto": "recto",
        "tiro alto": "tiro alto",
        "talle alto": "tiro alto",

        # Largos y Formatos
        "midi": "midi",
        "maxi": "maxi",
        "palazzo": "palazzo",
        "flare": "flare",
        "culotte": "culotte",
        "tipo pareo": "tipo pareo",

        # Colores
        "turquesa": "turquesa",
        "marrón": "marrón",
        "beige": "beige",
        "mostaza": "mostaza",
        "blanco crema": "blanco crema",
        "azul": "azul",
        "negro": "negro",
        "amarillo": "amarillo",
        "gris verde": "gris verde",

        "cuello barco ancho": "cuello barco ancho",
        "cuello camisero abierto": "cuello camisero abierto",
        "cuello de encaje": "cuello de encaje",
        "cuello con botones": "cuello con botones",
        "cuello plisado": "cuello plisado",
        "cuello drapeado amplio": "cuello drapeado amplio",

        # Mangas
        "manga 3/4 acampanada": "manga 3/4 acampanada",
        "manga francesa ajustada": "manga francesa ajustada",
        "manga larga globo": "manga larga globo",
        "manga corta con volante": "manga corta con volante",
        "manga corta plisada": "manga corta plisada",
        "manga larga con abertura lateral": "manga larga con abertura lateral",
        "manga con lazada": "manga con lazada",
        "manga con puño elástico": "manga con puño elástico",

        # Tejidos y Materiales
        "lana": "lana",
        "cachemira": "cachemira",
        "organza": "organza",
        "chiffon": "chiffon",
        "tweed": "tweed",
        "cuero": "cuero",
        "ante": "ante",
        "lycra": "lycra",
        "malla": "malla",

        # Estampados
        "estampado de lunares": "estampado de lunares",
        "estampado de cuadros": "estampado de cuadros",
        "estampado tartán": "estampado tartán",
        "estampado paisley": "estampado paisley",
        "estampado camuflaje": "estampado camuflaje",

        # Detalles y Acabados
        "bordado artesanal": "bordado artesanal",
        "detalle con tachuelas": "detalle con tachuelas",
        "detalle con cadenas": "detalle con cadenas",
        "detalle con paillettes": "detalle con paillettes",
        "pinzas en cintura": "pinzas en cintura",
        "corte asimétrico": "corte asimétrico",
        "bajo redondeado": "bajo redondeado",

        # Cierres
        "cierre de cremallera lateral": "cierre de cremallera lateral",
        "cierre de botones delanteros": "cierre de botones delanteros",
        "cierre de corchetes": "cierre de corchetes",
        "cierre de broche": "cierre de broche",
        "cierre ajustable con cordón": "cierre ajustable con cordón",

        # Siluetas y Cortes
        "corte evasé": "corte evasé",
        "corte globo": "corte globo",
        "corte trapecio": "corte trapecio",
        "corte imperio": "corte imperio",

        # Largos y Formatos
        "mini": "mini",
        "rodilla": "rodilla",
        "hasta el tobillo": "hasta el tobillo",
        "largo por encima del tobillo": "largo por encima del tobillo",

        # Colores
        "borgoña": "borgoña",
        "burdeos": "burdeos",
        "mostaza oscuro": "mostaza oscuro",
        "verde oliva": "verde oliva",
        "azul marino": "azul marino",
        "coral": "coral",
    }

    # Normalización de variantes comunes
    # Diccionario de sinónimos normalizados
    sinonimos = {
        # 🔹 Tipos de prenda
        "camiseta": ["playera", "remera", "t-shirt"],
        "jersey": ["suéter", "pulóver"],
        "abrigo": ["chaquetón", "sobretodo"],
        "chaleco": ["gilet"],
        "traje": ["terno"],
        "pantalón": ["pantalones", "pantalón largo"],

        # 🔹 Mangas
        "manga larga": ["manga extendida"],
        "manga corta": ["manga breve"],
        "manga francesa": ["manga tres cuartos"],
        "manga farol": ["manga globo"],
        "manga ranglán": ["manga raglán"],
        "sin mangas": ["sisa"],

        # 🔹 Cuellos
        "cuello redondo": ["escote redondo"],
        "cuello pico": ["cuello en v", "escote en v"],
        "cuello barco": ["escote barco"],
        "cuello cisne": ["cuello tortuga"],
        "cuello sweetheart": ["escote corazón"],
        "cuello palabra de honor": ["escote palabra de honor"],

        # 🔹 Tejidos
        "denim": ["mezclilla"],
        "algodón": ["cotton"],
        "seda": ["silk"],
        "lana": ["wool"],
        "poliéster": ["polyester"],
        "elastano": ["spandex", "lycra"],

        # 🔹 Detalles
        "volantes": ["fruncidos", "pliegues decorativos"],
        "encaje": ["puntilla"],
        "lentejuelas": ["pailletes"],
        "transparencias": ["efecto transparente"],

        # 🔹 Estilo
        "boho": ["bohemio"],
        "casual": ["informal"],
        "elegante": ["sofisticado", "chic"],
        "sport": ["deportivo"],
        "oversized": ["amplio", "extra grande"],
        "cropped": ["corto", "recortado"],

        # 🔹 Corte
        "corte evasé": ["línea a"],
        "corte entallado": ["ajustado", "slim fit"],
        "corte holgado": ["flojo", "sueltito"],
        "corte recto": ["línea recta"],

        # 🔹 Colores
        "negro": ["black"],
        "blanco": ["white"],
        "rojo": ["red"],
        "azul": ["blue"],
        "verde": ["green"],
        "amarillo": ["yellow"],
        "rosa": ["pink"],
        "morado": ["lila", "violeta"],
        "naranja": ["anaranjado"],
        "beige": ["crema", "arena"],
        "gris": ["plomo", "gray"],
        "marrón": ["café", "chocolate"],
        "dorado": ["oro"],
        "plateado": ["plata"],

        # Cuellos
        "cuello barco ancho": ["cuello barco grande", "cuello bote ancho"],
        "cuello camisero abierto": ["cuello camisa abierto", "cuello estilo camisa"],
        "cuello de encaje": ["cuello con encaje", "cuello encaje"],
        "cuello con botones": ["cuello abotonado", "cuello con abotonadura"],
        "cuello plisado": ["cuello con pliegues", "cuello fruncido"],
        "cuello drapeado amplio": ["cuello drapeado", "cuello con drapeado"],

        # Mangas
        "manga 3/4 acampanada": ["manga tres cuartos acampanada", "manga 3/4 campana"],
        "manga francesa ajustada": ["manga francesa ceñida", "manga francesa estrecha"],
        "manga larga globo": ["manga globo larga", "manga bombon larga"],
        "manga corta con volante": ["manga corta avolantada", "manga corta con volantes"],
        "manga corta plisada": ["manga corta con pliegues", "manga corta fruncida"],
        "manga larga con abertura lateral": ["manga larga con corte lateral", "manga larga con abertura"],
        "manga con lazada": ["manga con lazo", "manga con atadura"],
        "manga con puño elástico": ["manga con puño fruncido", "manga con puño ajustado"],

        # Tejidos y Materiales
        "lana": ["tejido lana", "lana natural"],
        "cachemira": ["cashmere", "lana cachemir"],
        "organza": ["tela organza", "tejido organza"],
        "chiffon": ["chifón", "tejido chiffon"],
        "tweed": ["tejido tweed", "tela tweed"],
        "cuero": ["piel", "piel sintética"],
        "ante": ["gamuza", "ante sintético"],
        "lycra": ["tejido lycra", "material lycra"],
        "malla": ["tejido malla", "red malla"],

        # Estampados
        "estampado de lunares": ["print de lunares", "motivo de lunares"],
        "estampado de cuadros": ["print de cuadros", "motivo de cuadros"],
        "estampado tartán": ["print tartán", "motivo tartán"],
        "estampado paisley": ["print paisley", "motivo paisley"],
        "estampado camuflaje": ["print camuflaje", "motivo camuflaje"],

        # Detalles y Acabados
        "bordado artesanal": ["bordado hecho a mano", "bordado manual"],
        "detalle con tachuelas": ["adornos con tachuelas", "decoración con tachuelas"],
        "detalle con cadenas": ["adornos con cadenas", "decoración con cadenas"],
        "detalle con paillettes": ["detalle con lentejuelas", "decoración con paillettes"],
        "pinzas en cintura": ["pliegues en cintura", "pinzas laterales"],
        "corte asimétrico": ["diseño asimétrico", "corte irregular"],
        "bajo redondeado": ["dobladillo redondo", "bajo curvo"],

        # Cierres
        "cierre de cremallera lateral": ["cremallera lateral", "cierre lateral"],
        "cierre de botones delanteros": ["botones frontales", "cierre frontal con botones"],
        "cierre de corchetes": ["broches", "cierres tipo corchete"],
        "cierre de broche": ["broche", "cierres con broche"],
        "cierre ajustable con cordón": ["ajuste con cordón", "cierre con cordón ajustable"],

        # Siluetas y Cortes
        "corte evasé": ["corte en evasé", "falda evasé"],
        "corte globo": ["corte tipo globo", "falda globo"],
        "corte trapecio": ["corte en trapecio", "falda trapecio"],
        "corte imperio": ["corte estilo imperio", "vestido imperio"],

        # Largos y Formatos
        "mini": ["corto", "miniatura"],
        "rodilla": ["largo a la rodilla", "a la altura de rodilla"],
        "hasta el tobillo": ["largo hasta el tobillo", "largo tobillo"],
        "largo por encima del tobillo": ["largo sobre tobillo", "largo justo encima del tobillo"],

        # Colores
        "borgoña": ["burdeos", "vino"],
        "burdeos": ["borgoña", "vino"],
        "mostaza oscuro": ["mostaza intenso", "mostaza oscuro"],
        "verde oliva": ["verde militar", "verde oliva oscuro"],
        "azul marino": ["navy", "azul oscuro"],
        "coral": ["naranja rosado", "rosa coral"],
    }

    # Step 4: Normalize synonyms function
    def normalizar_sinonimos(texto):
        for canonico, variantes in sinonimos.items():
            for variante in variantes:
                texto = re.sub(rf'\b{re.escape(variante)}\b', canonico, texto, flags=re.IGNORECASE)
        return texto

    # Step 5: Basic text cleaning
    def limpiar_texto(texto):
        if not isinstance(texto, str):
            return ""
        texto = texto.lower()
        texto = re.sub(r'\b(y|o|de|con|el|la|los|las|un|una|unos|unas)\b', '', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        return texto
    
    # Step 6: Entity extraction setup
    entidades_def = [
        # 🔹 Tipos de Prenda
        ("abrigo", "TIPO_PRENDA"),
        ("anorak", "TIPO_PRENDA"),
        ("blazer", "TIPO_PRENDA"),
        ("blusa", "TIPO_PRENDA"),
        ("body", "TIPO_PRENDA"),
        ("camisa", "TIPO_PRENDA"),
        ("camiseta", "TIPO_PRENDA"),
        ("chal", "TIPO_PRENDA"),
        ("chaleco", "TIPO_PRENDA"),
        ("chubasquero", "TIPO_PRENDA"),
        ("cárdigan", "TIPO_PRENDA"),
        ("falda", "TIPO_PRENDA"),
        ("gabardina", "TIPO_PRENDA"),
        ("jersey", "TIPO_PRENDA"),
        ("kimono", "TIPO_PRENDA"),
        ("mono", "TIPO_PRENDA"),
        ("pantalón", "TIPO_PRENDA"),
        ("peto", "TIPO_PRENDA"),
        ("top", "TIPO_PRENDA"),
        ("traje", "TIPO_PRENDA"),
        ("vestido", "TIPO_PRENDA"),

        # 🔹 Mangas
        ("manga acampanada", "MANGA"),
        ("manga abombada", "MANGA"),
        ("manga abullonada", "MANGA"),
        ("manga ajustada", "MANGA"),
        ("manga asimétrica", "MANGA"),
        ("manga caida", "MANGA"),
        ("manga campana", "MANGA"),
        ("manga capa", "MANGA"),
        ("manga corta", "MANGA"),
        ("manga con abertura", "MANGA"),
        ("manga con bordado", "MANGA"),
        ("manga con nudo", "MANGA"),
        ("manga con puño", "MANGA"),
        ("manga con volante", "MANGA"),
        ("manga desmontable", "MANGA"),
        ("manga doble", "MANGA"),
        ("manga estructurada", "MANGA"),
        ("manga extralarga", "MANGA"),
        ("manga farol", "MANGA"),
        ("manga francesa", "MANGA"),
        ("manga fruncida", "MANGA"),
        ("manga globo", "MANGA"),
        ("manga holgada", "MANGA"),
        ("manga jamón", "MANGA"),
        ("manga kimono", "MANGA"),
        ("manga larga", "MANGA"),
        ("manga larga ajustada", "MANGA"),
        ("manga murciélago", "MANGA"),
        ("manga plisada", "MANGA"),
        ("manga ranglán", "MANGA"),  # Variante de "raglán"
        ("manga transparente", "MANGA"),
        ("sin mangas", "MANGA"),

        # 🔹 Cuellos
        ("cuello asimétrico", "CUELLO"),
        ("cuello alto", "CUELLO"),
        ("cuello barco", "CUELLO"),
        ("cuello bebé", "CUELLO"),
        ("cuello caja", "CUELLO"),
        ("cuello camisero", "CUELLO"),
        ("cuello chimenea", "CUELLO"),
        ("cuello cisne", "CUELLO"),
        ("cuello con abertura", "CUELLO"),
        ("cuello con lazo", "CUELLO"),
        ("cuello con volante", "CUELLO"),
        ("cuello cuadrado", "CUELLO"),
        ("cuello cuadrado profundo", "CUELLO"),
        ("cuello cruzado", "CUELLO"),
        ("cuello drapeado", "CUELLO"),
        ("cuello en v", "CUELLO"),
        ("cuello halter", "CUELLO"),
        ("cuello ilusión", "CUELLO"),
        ("cuello mao", "CUELLO"),
        ("cuello perkins", "CUELLO"),
        ("cuello pico", "CUELLO"),
        ("cuello palabra de honor", "CUELLO"),
        ("cuello polo", "CUELLO"),
        ("cuello redondo", "CUELLO"),
        ("cuello sweetheart", "CUELLO"),
        ("cuello smoking", "CUELLO"),
        ("cuello tortuga", "CUELLO"),
        ("cuello volante", "CUELLO"),

        # 🔹 Tejidos
        ("algodón", "TEJIDO"),
        ("seda", "TEJIDO"),
        ("lino", "TEJIDO"),
        ("poliéster", "TEJIDO"),
        ("lana", "TEJIDO"),
        ("encaje", "TEJIDO"),
        ("satén", "TEJIDO"),
        ("terciopelo", "TEJIDO"),
        ("denim", "TEJIDO"),
        ("viscosa", "TEJIDO"),
        ("elastano", "TEJIDO"),
        ("punto", "TEJIDO"),
        ("chiffon", "TEJIDO"),
        ("gasa", "TEJIDO"),

        # 🔹 Detalles
        ("bordado", "DETALLE"),
        ("volantes", "DETALLE"),
        ("encaje", "DETALLE"),
        ("lentejuelas", "DETALLE"),
        ("perlas", "DETALLE"),
        ("pedrería", "DETALLE"),
        ("botones decorativos", "DETALLE"),
        ("aberturas laterales", "DETALLE"),
        ("transparencias", "DETALLE"),
        ("nudo frontal", "DETALLE"),

        # 🔹 Estilo
        ("boho", "ESTILO"),
        ("casual", "ESTILO"),
        ("elegante", "ESTILO"),
        ("formal", "ESTILO"),
        ("informal", "ESTILO"),
        ("minimalista", "ESTILO"),
        ("romántico", "ESTILO"),
        ("sport", "ESTILO"),
        ("urban", "ESTILO"),
        ("vintage", "ESTILO"),
        ("oversized", "ESTILO"),
        ("cropped", "ESTILO"),

        # 🔹 Corte
        ("corte evasé", "CORTE"),
        ("corte recto", "CORTE"),
        ("corte entallado", "CORTE"),
        ("corte holgado", "CORTE"),
        ("corte asimétrico", "CORTE"),
        ("corte acampanado", "CORTE"),
        ("corte midi", "CORTE"),
        ("corte mini", "CORTE"),
        ("corte maxi", "CORTE"),

        # 🔹 Colores
        ("negro", "COLOR"),
        ("blanco", "COLOR"),
        ("rojo", "COLOR"),
        ("azul", "COLOR"),
        ("verde", "COLOR"),
        ("amarillo", "COLOR"),
        ("rosa", "COLOR"),
        ("morado", "COLOR"),
        ("naranja", "COLOR"),
        ("beige", "COLOR"),
        ("gris", "COLOR"),
        ("marrón", "COLOR"),
        ("dorado", "COLOR"),
        ("plateado", "COLOR"),
        ("chaqueta", "TIPO_PRENDA"),
        ("cardigan", "TIPO_PRENDA"),
        ("gabardina", "TIPO_PRENDA"),
        ("camisón", "TIPO_PRENDA"),
        ("polo", "TIPO_PRENDA"),

        # Cuellos (añadidos nuevos)
        ("cuello barco ancho", "CUELLO"),
        ("cuello camisero abierto", "CUELLO"),
        ("cuello de encaje", "CUELLO"),
        ("cuello con botones", "CUELLO"),
        ("cuello plisado", "CUELLO"),
        ("cuello drapeado amplio", "CUELLO"),

        # Mangas (añadidos nuevos)
        ("manga 3/4 acampanada", "MANGA"),
        ("manga francesa ajustada", "MANGA"),
        ("manga larga globo", "MANGA"),
        ("manga corta con volante", "MANGA"),
        ("manga corta plisada", "MANGA"),
        ("manga larga con abertura lateral", "MANGA"),
        ("manga con lazada", "MANGA"),
        ("manga con puño elástico", "MANGA"),

        # Tejidos y materiales (añadidos nuevos)
        ("lana", "TEJIDO"),
        ("cachemira", "TEJIDO"),
        ("organza", "TEJIDO"),
        ("chiffon", "TEJIDO"),
        ("tweed", "TEJIDO"),
        ("cuero", "TEJIDO"),
        ("ante", "TEJIDO"),
        ("lycra", "TEJIDO"),
        ("malla", "TEJIDO"),

        # Estampados (añadidos nuevos)
        ("estampado de lunares", "ESTAMPADO"),
        ("estampado de cuadros", "ESTAMPADO"),
        ("estampado tartán", "ESTAMPADO"),
        ("estampado paisley", "ESTAMPADO"),
        ("estampado camuflaje", "ESTAMPADO"),

        # Detalles y acabados (añadidos nuevos)
        ("bordado artesanal", "DETALLE"),
        ("detalle con tachuelas", "DETALLE"),
        ("detalle con cadenas", "DETALLE"),
        ("detalle con paillettes", "DETALLE"),
        ("pinzas en cintura", "DETALLE"),
        ("corte asimétrico", "DETALLE"),
        ("bajo redondeado", "DETALLE"),

        # Cierres (añadidos nuevos)
        ("cierre de cremallera lateral", "CIERRE"),
        ("cierre de botones delanteros", "CIERRE"),
        ("cierre de corchetes", "CIERRE"),
        ("cierre de broche", "CIERRE"),
        ("cierre ajustable con cordón", "CIERRE"),

        # Siluetas y cortes (añadidos nuevos)
        ("corte evasé", "CORTE"),
        ("corte globo", "CORTE"),
        ("corte trapecio", "CORTE"),
        ("corte imperio", "CORTE"),

        # Largos y formatos (añadidos nuevos)
        ("mini", "LARGO"),
        ("rodilla", "LARGO"),
        ("hasta el tobillo", "LARGO"),
        ("largo por encima del tobillo", "LARGO"),

        # Colores (añadidos nuevos)
        ("borgoña", "COLOR"),
        ("burdeos", "COLOR"),
        ("mostaza oscuro", "COLOR"),
        ("verde oliva", "COLOR"),
        ("azul marino", "COLOR"),
        ("coral", "COLOR"),
    ]
    
    # Initialize spaCy
    try:
        nlp = spacy.load("es_core_news_sm")
    except OSError:
        st.error("❌ Spanish language model not found. Please install: python -m spacy download es_core_news_sm")
        return None
    
    # Setup matcher
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    # Group terms by label
    terminos_por_etiqueta = {}
    for term, label in entidades_def:
        terminos_por_etiqueta.setdefault(label, []).append(term)

    # Add patterns to matcher
    for label, terms in terminos_por_etiqueta.items():
        patterns = [nlp.make_doc(term.lower()) for term in terms]
        matcher.add(label, patterns)

    # Step 7: Entity extraction function
    def extraer_entidades(texto):
        if pd.isna(texto):
            return {label: [] for label in terminos_por_etiqueta.keys()}
        
        texto_limpio = limpiar_texto(texto)
        texto_normalizado = normalizar_sinonimos(texto_limpio)
        doc = nlp(texto_normalizado)
        
        resultado = {label: [] for label in terminos_por_etiqueta.keys()}
        matches = matcher(doc)
        
        for match_id, start, end in matches:
            label = nlp.vocab.strings[match_id]
            span = doc[start:end].text
            
            # Clean prefixes for neck/sleeve
            if label == "CUELLO" and span.startswith("cuello "):
                span = span.replace("cuello ", "")
            elif label == "MANGA" and span.startswith("manga "):
                span = span.replace("manga ", "")
            
            if span not in resultado[label]:
                resultado[label].append(span)
        
        return resultado

    # Step 8: Apply preprocessing
    st.info("🔄 Processing descriptions...")
    
    # Filter and prepare DataFrame
    df_clean = df_filtered.dropna(subset=['fashion_main_description_1']).reset_index(drop=True)
    df_unique = df_clean.drop_duplicates(subset=['fashion_main_description_1']).reset_index(drop=True)

    # Apply entity extraction
    resultados = df_unique['fashion_main_description_1'].apply(extraer_entidades)

    # Create columns for each entity type
    for etiqueta in terminos_por_etiqueta.keys():
        df_unique[etiqueta] = resultados.apply(lambda x: ', '.join(x[etiqueta]) if x[etiqueta] else '')

    # Step 9: Prepare final output
    # Select columns for dashboard
    output_columns = ['provider_ref', 'fashion_main_description_1', 'MANGA', 'CUELLO', 'TEJIDO', 'DETALLE', 'ESTILO', 'CORTE']
    available_columns = [col for col in output_columns if col in df_unique.columns]

    df_final = df_unique[available_columns].copy()

    # Rename provider_ref to Código único for dashboard compatibility
    if 'provider_ref' in df_final.columns:
        df_final = df_final.rename(columns={'provider_ref': 'Código único'})

    st.success(f"✅ Processing complete! Final shape: {df_final.shape}")

    return df_final

def get_processed_descriptions(uploaded_files):
    """
    Main function to be called from dashboard
    """
    if not uploaded_files:
        return None
    
    return preprocess_description_files(uploaded_files) 