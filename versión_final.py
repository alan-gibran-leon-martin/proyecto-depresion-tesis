import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path
from cryptography.fernet import Fernet
from supabase import create_client, Client

# Dependencias NLTK
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
carpetaactual= Path(__file__).resolve().parent
# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS
# ==========================================
st.set_page_config(
    page_title="Sistema de Evaluación de Depresión",
    initial_sidebar_state="collapsed",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
            border-right: 1px solid #e0e0e0;
        }
        .step-inactive {
            color: #6c757d;
            padding: 4px 0px;
            font-size: 14px;
        }
        .step-active {
            color: #0d6efd;
            font-weight: bold;
            padding: 4px 0px;
            font-size: 14px;
        }
        .stButton button {
            background-color: #0d6efd;
            color: white;
            border-radius: 6px;
            padding: 0.5rem 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# CONFIGURACIÓN DE SUPABASE Y FERNET
# ==========================================
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "https://wotijdutbzbiwgdpmjfs.supabase.co/rest/v1/")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "TU_SUPABASE_KEY")

@st.cache_resource
@st.cache_resource
def init_supabase() -> Client:
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Error al conectar con Supabase: {e}")
        return None

supabase = init_supabase()

@st.cache_resource
def get_fernet_cipher():
    key = st.secrets["FERNET_KEY"]
    return Fernet(key.encode() if isinstance(key, str) else key)

cipher_suite = get_fernet_cipher()
@st.cache_resource
# CARGA DE METADATOS


@st.cache_data
def cargar_metadatos():
    try:
        edad_df = pd.read_csv(carpetaactual / "metadatos" / "porcentaje_edad.csv")
        sexo_df = pd.read_csv(carpetaactual / "metadatos" / "porcentajesexo.csv")
        escolaridad_df = pd.read_csv(carpetaactual / "metadatos" / "porcentajeeductaivo.csv", encoding="latin-1")
        orientacion_df = pd.read_csv(carpetaactual / "metadatos" / "orientacion_sexual.csv")
        return edad_df, sexo_df, escolaridad_df, orientacion_df
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

edad_df, sexo_df, escolaridad_df, orientacion_df = cargar_metadatos()

# ==========================================
# ESTADO DE NAVEGACIÓN
# ==========================================
if 'screen' not in st.session_state:
    st.session_state.screen = 1
if 'data' not in st.session_state:
    st.session_state.data = {}
if 'asq_triggered' not in st.session_state:
    st.session_state.asq_triggered = False

# ==========================================
# BARRA LATERAL
# ==========================================
pantallas_nombres = [
    "1. Bienvenida",
    "2. Consentimiento",
    "3. Datos personales",
    "4. Cuestionario sociodemográfico",
    "5. Información adicional",
    "6. BDI",
    "7. Riesgo de suicidio",
    "8. Resultados",
    "9. Protocolo de derivación (depresión)",
    "10. Protocolo de derivación (suicidio)"
]

with st.sidebar:
    st.image("https://img.icons8.com/color/96/brain.png", width=60)
    st.markdown("### Navegación")
    for i, nombre in enumerate(pantallas_nombres, 1):
        if st.session_state.screen == i:
            st.markdown(f'<div class="step-active">▶ {nombre}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="step-inactive">  {nombre}</div>', unsafe_allow_html=True)

col_top1, col_top2 = st.columns([10, 2])
with col_top2:
    if st.button("Salir", use_container_width=True):
        nombre_usuario= st.session_state.data.get('nombre', 'participante')
        st.warning(f"⚠️ ¿Está seguro que desea salir? {nombre_usuario} si continuas, la información ingresada no se guardará.")
        if st.button("Sí, salir y reiniciar"):
            st.session_state.clear()
            st.rerun()

screen = st.session_state.screen

# ------------------------------------------
# PANTALLA 1: BIENVENIDA
# ------------------------------------------
if screen == 1:
    st.markdown("<h1 style='text-align: center;'>🧠 Sistema de Evaluación de Depresión</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: gray;'>Bienvenido </h3>", unsafe_allow_html=True)
    st.write("""
    Esta aplicación combina análisis estadísticos (teorema de Bayes) y procesamiento automático de lenguaje natural (NLP) 
    para estimar indicadores de depresión de forma automatizada. 
    Su participación es estrictamente voluntaria y sus datos serán tratados con absoluta confidencialidad.
    """)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Continuar", use_container_width=True):
            st.session_state.screen = 2
            st.rerun()

# ------------------------------------------
# PANTALLA 2: CONSENTIMIENTO INFORMADO
# ------------------------------------------
elif screen == 2:
    st.subheader("Consentimiento informado")
    st.write("Antes de continuar, le pedimos que lea y acepte el siguiente consentimiento informado:")
    with st.container(height=300):
        st.markdown("""
## Consentimiento informado.  
### Evaluación exploratoria de indicadores lingüísticos y variables sociodemográficas asociados a la depresión  
Se le invita a participar en este estudio de investigación. Antes de tomar una decisión, es importante que usted conozca y comprenda la siguiente información. Por favor, pregunte sobre cualquier duda o información que desee conocer.  
Su consentimiento para participar se dará por entendido al firmar este documento.  
**Acerca de los motivos de realización de este estudio**  
La depresión es un trastorno mental común que afecta aproximadamente al 4% de la población mundial (Organización Mundial de la Salud, 2023). En México, los recursos destinados a la atención en salud mental son limitados, y el diagnóstico actual se basa principalmente en criterios clínicos y autoreporte, lo que presenta limitaciones en términos de objetividad. El presente estudio busca explorar el uso de herramientas computacionales para el análisis del lenguaje como una vía complementaria para la identificación de indicadores asociados a la depresión, con fines exclusivamente de investigación.  
**Procedimiento**  
Si usted acepta participar, se le solicitará completar un cuestionario que incluye:  
* Una pregunta abierta en la que describirá cómo se ha sentido emocionalmente durante las últimas dos semanas  
* Preguntas sociodemográficas (edad, sexo, nivel educativo, ocupación)  
* Un cuestionario estandarizado sobre síntomas de depresión (Inventario de Depresión de Beck-II).  
* Un cuestionario breve sobre suicidio (ASQ).  
La participación es individual y autoadministrada, y tendrá una duración aproximada de 10 a 15 minutos.  
**Riesgos**  
La participación en este estudio implica riesgos mínimos. Algunas preguntas podrían generar incomodidad emocional leve al abordar temas relacionados con el estado de ánimo.  
En caso de que durante su participación se presente malestar significativo, se recomienda acudir a un profesional de la salud mental. Si requiere atención inmediata, puede contactar a la Línea de la Vida (800 911 2000) o acudir a su unidad de salud más cercana.  
**Beneficios**  
No se obtendrán beneficios directos por participar en este estudio. Sin embargo, su participación contribuirá al desarrollo de conocimiento en el área de la salud mental y al diseño de herramientas de apoyo para su evaluación.  
**Privacidad y manejo de información.**  
La información proporcionada será tratada de manera confidencial y anónima. No se recopilarán datos que permitan identificar directamente a los participantes (como nombre, domicilio o correo electrónico).  
Los datos serán almacenados en una base de datos protegida mediante mecanismos de seguridad digital. El acceso a la información estará restringido únicamente al investigador responsable del estudio.  
Los datos serán utilizados exclusivamente con fines académicos y de investigación. En ningún caso se publicará información individual que permita identificar a los participantes. Los datos serán eliminados una vez que dejen de ser necesarios para los fines del estudio.  
El tratamiento de los datos se realizará conforme a lo establecido en la Ley General de Protección de Datos Personales en Posesión de Sujetos Obligados.  
**Suspensión o término de la participación.**  
Usted puede negarse a participar en este estudio o terminar su participación en cualquier momento, sin necesidad de dar explicaciones y sin consecuencias negativas. Si decide retirarse, sus datos serán eliminados de la base de datos y no se utilizarán para los fines del estudio.  
**Responsable del estudio.**  
El investigador responsable del estudio es:  
Alan Gibran Leon Martin  
Facultad de Psicología, Universidad Nacional Autónoma de México (UNAM)  
Correo electrónico: alangibranleonmartin@gmail.com  
Puede contactar al investigador responsable si tiene alguna pregunta, duda o inquietud sobre el estudio, o si desea retirar su participación.  
**Declaración de consentimiento**  
He leído y comprendido la información proporcionada. Se me ha explicado el propósito del estudio, los procedimientos, los posibles riesgos y beneficios, así como mis derechos como participante.  
Por favor, firme este formulario de consentimiento sólo si ha comprendido plenamente la naturaleza y el procedimiento de este estudio descritos en la hoja de información, y únicamente si no tiene dudas y ha entendido todos sus derechos como participante.  

        """)
    consent = st.checkbox("Acepto los términos del consentimiento informado")
    consent2 = st.checkbox("Soy mayor de 18 años y doy mi consentimiento para participar en este estudio. ")
    consent3 = st.checkbox("Certifico que he leído y comprendido la información anterior, que he recibido respuestas satisfactorias a mis preguntas con respecto a este proyecto, y que se me ha informado que tengo la libertad de retirar mi consentimiento y poner fin a mi participación en el proyecto en cualquier momento. ")
    consent4 = st.checkbox("Por la presente doy mi consentimiento para participar en este proyecto con el entendimiento de que dicho consentimiento no conlleva la renuncia a ninguno de mis derechos legales, ni tampoco exime a los investigadores o cualquier agente de LA MISMA de la responsabilidad por negligencia.")
    consent5= st.checkbox("Acepto que la información que proporcione, incluyendo mis respuestas al cuestionario y el texto que escriba, sea almacenada de forma segura y utilizada para el entrenamiento, ajuste y prueba del modelo computacional y del sistema general desarrollado en este proyecto. Se me ha informado que esta información será tratada de forma anónima y confidencial, que no se utilizará para fines distintos a los establecidos en este estudio, y que podré solicitar su eliminación en cualquier momento sin que ello implique consecuencias negativas para mí.")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Continuar", use_container_width=True):
            if consent and consent2 and consent3 and consent4 and consent5:
                st.session_state.screen = 3
                st.rerun()
            else:
                st.error("Debe aceptar el consentimiento informado para continuar.")

# ------------------------------------------
# PANTALLA 3: DATOS PERSONALES Y CONTACTO
# ------------------------------------------
elif screen == 3:
    st.subheader("Tus datos personales y de contacto")
    st.write("Indícanos cómo te gustaría que nos refiramos a ti y un medio de contacto (correo o teléfono) para asociar tu registro.")
    
    ref_option = st.write("¿Cómo prefieres que nos refiramos a ti?")
    nombre_o_seudonimo = st.text_input("Ingresa tu nombre o seudónimo") 
        
    contacto = st.text_input("Correo electrónico o teléfono de contacto", placeholder="ejemplo@correo.com o 5512345678")
    
    st.info("🔒 Tu contacto se almacenará cifrado simétricamente y vinculado de forma independiente a tu registro para evitar duplicidades.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Continuar", use_container_width=True):
            if nombre_o_seudonimo.strip() != "" and contacto.strip() != "":
                st.session_state.data['nombre'] = nombre_o_seudonimo
                st.session_state.data['contacto'] = contacto
                st.session_state.screen = 4
                st.rerun()
            else:
                st.error("Por favor, completa tu nombre/seudónimo y tu medio de contacto.")

# ------------------------------------------
# PANTALLA 4: CUESTIONARIO SOCIODEMOGRÁFICO Y PREGUNTA ABIERTA
# ------------------------------------------
elif screen == 4:
    st.subheader("Cuestionario sociodemográfico")
    col_a, col_b = st.columns(2)
    with col_a:
        edad_input = st.number_input("Edad", min_value=18, max_value=110, value=25)
        sexo_input = st.selectbox("Sexo asignado al nacer", ["masculino", "Femenino"])
    with col_b:
        escolaridad_input = st.selectbox("¿Cuál es tu nivel educativo?", [
            "ningún año cursado", "prescolar", "primaria", "secundaria", "bachillerato", 
            "normal básica", "estudios técnicos o comerciales con primaria terminada", 
            "estudios técnicos o comerciales con secundaria terminada", 
            "estudios técnicos o comerciales con preparatoria terminada", 
            "normal de licenciatura", "licenciatura profesional", "maestria", "doctorado"
        ])
        orientacion_input = st.selectbox("¿Cuál es tu orientación sexual?", ["heterosexual", "no-heterosexual"])
    
    antecedente_input = st.selectbox("¿Alguno de tus padres fue diagnosticado con depresión?", ["sin-antecedentes", "con-diagnostico"])
    
    st.markdown("---")
    st.write("**Pregunta abierta**")
    pregunta_abierta = st.text_area("En las últimas dos semanas incluyendo el día de hoy describa como se ha sentido (opcional):", placeholder="Escribe aquí...")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Continuar", use_container_width=True):
            st.session_state.data['edad'] = int(edad_input)
            st.session_state.data['sexo'] = sexo_input
            st.session_state.data['escolaridad'] = escolaridad_input
            st.session_state.data['orientacion'] = orientacion_input
            st.session_state.data['antecedente'] = antecedente_input
            st.session_state.data['pregunta_abierta'] = pregunta_abierta if pregunta_abierta else "sin respuesta"
            st.session_state.screen = 5
            st.rerun()

# ------------------------------------------
# PANTALLA 5: INFORMACIÓN ADICIONAL
# ------------------------------------------
elif screen == 5:
    st.subheader("Información importante")
    st.info("""
    A continuación, te haremos algunas preguntas adicionales que pueden parecer repetitivas, pero son necesarias para complementar la evaluación y mejorar la precisión de los resultados.
    """)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Continuar", use_container_width=True):
            st.session_state.screen = 6
            st.rerun()

# ------------------------------------------
# PANTALLA 6: BDI-II
# ------------------------------------------
elif screen == 6:
    st.subheader("Inventario de Depresión de Beck (BDI)")
    st.write("Instrucciones: Este cuestionario consta de 21 grupos de afirmaciones. Por favor, lea con atención cada uno de ellos cuidadosamente. Luego elija uno de cada grupo, el que mejor describa el modo como se ha sentido las últimas dos semanas, incluyendo el día de hoy.Marque con un círculo el número correspondiente al enunciado elegido Si varios enunciados de un mismo grupo le parecen igualmente apropiados, marque el número más alto. Verifique que no haya elegido más de uno por grupo, incluyendo el ítem 16 (cambios en los hábitos de Sueño) y el ítem 18 (cambios en el apetito),")
    
    bdi_preguntas = [
        ("1. Tristeza", ["No me siento triste", "Me siento triste gran parte del tiempo", "Me siento triste todo el tiempo", "Estoy tan infeliz o triste que no puedo soportarlo"]),
        ("2. Pesimismo", ["No estoy desanimado respecto al futuro", "Siento que tengo más desánimo respecto al futuro que antes", "No espero que las cosas funcionen para mí", "Siento que el futuro es desesperanzador y que solo empeorará"]),
        ("3. Fracaso", ["No me siento como un fracasado", "He fracasado más de lo que debiera", "Cuando miro atrás, veo muchos fracasos", "Siento que soy un fracasado total como persona"]),
        ("4. Pérdida de Placer", ["Obtengo tanto placer como siempre por las cosas de las que disfruto", "No disfruto tanto de las cosas como solía hacerlo", "Obtengo muy poco placer de las cosas que solía disfrutar", "No puedo obtener ningún placer de las cosas de las que solía disfrutar"]),
        ("5. Sentimientos de Culpa", ["No me siento particularmente culpable", "Me siento culpable respecto de varias cosas que he hecho o que debería haber hecho", "Me siento bastante culpable la mayor parte del tiempo", "Me siento culpable todo el tiempo"]),
        ("6. Sentimientos de Castigo", ["No siento que esté siendo castigado", "Siento que tal vez pueda ser castigado", "Espero ser castigado", "Siento que estoy siendo castigado"]),
        ("7. Disconformidad con uno mismo", ["Siento acerca de mí lo mismo que siempre", "He perdido la confianza en mí mismo", "Estoy decepcionado conmigo mismo", "No me gusto a mí mismo"]),
        ("8. Autocrítica", ["No me critico ni me culpo más de lo habitual", "Estoy más crítico conmigo mismo de lo que solía estarlo", "Me critico a mí mismo por todos mis errores", "Me culpo a mí mismo por todo lo malo que sucede"]),
        ("9. Pensamientos o Deseos Suicidas", ["No tengo ningún pensamiento de matarme", "He tenido pensamientos de matarme, pero no lo haría", "Querría matarme", "Me mataría si tuviera la oportunidad de hacerlo"]),
        ("10. Llanto", ["No lloro más de lo que solía hacerlo", "Lloro más de lo que solía hacerlo", "Lloro por cualquier pequeñez", "Siento ganas de llorar pero no puedo"]),
        ("11. Agitación", ["No estoy más inquieto o tenso que lo habitual", "Me siento más inquieto o tenso que lo habitual", "Estoy tan inquieto o agitado que me es difícil quedarme quieto", "Estoy tan inquieto o agitado que tengo que estar siempre en movimiento o haciendo algo"]),
        ("12. Pérdida de Interés", ["No he perdido el interés en otras actividades o personas", "Estoy menos interesado que antes en otras personas o cosas", "He perdido casi todo el interés en otras personas o cosas", "Me es difícil interesarme por algo"]),
        ("13. Indecisión", ["Tomo mis propias decisiones tan bien como siempre", "Me resulta más difícil que de costumbre tomar decisiones", "Encuentro mucha más dificultad que antes para tomar decisiones", "Tengo problemas para tomar cualquier decisión"]),
        ("14. Desvalorización", ["No siento que yo no sea valioso", "No me considero a mí mismo tan valioso y útil como solía considerarme", "Me siento menos valioso cuando me comparo con otros", "Siento que no valgo nada"]),
        ("15. Pérdida de Energía", ["Tengo tanta energía como siempre", "Tengo menos energía que la que solía tener", "No tengo suficiente energía para hacer demasiado", "No tengo energía suficiente para hacer nada"]),
        ("16. Cambios en los Hábitos de Sueño", ["No he experimentado ningún cambio en mis hábitos de sueño", "Duermo un poco más que lo habitual", "Duermo un poco menos que lo habitual", "Duermo mucho más que lo habitual", "Duermo mucho menos que lo habitual", "Duermo la mayor parte del día", "Me despierto 1-2 horas más temprano y no puedo volver a dormirme"]),
        ("17. Irritabilidad", ["No estoy tan irritable que lo habitual", "Estoy más irritable que lo habitual", "Estoy mucho más irritable que lo habitual", "Estoy irritable todo el tiempo"]),
        ("18. Cambios en el Apetito", ["No he experimentado ningún cambio en mi apetito", "Mi apetito es un poco menor que lo habitual", "Mi apetito es un poco mayor que lo habitual", "Mi apetito es mucho menor que antes", "Mi apetito es mucho mayor que lo habitual", "No tengo apetito en absoluto", "Quiero comer todo el día"]),
        ("19. Dificultad de Concentración", ["Puedo concentrarme tan bien como siempre", "No puedo concentrarme tan bien como habitualmente", "Me es difícil mantener la mente en algo por mucho tiempo", "Encuentro que no puedo concentrarme en nada"]),
        ("20. Cansancio o Fatiga", ["No estoy más cansado o fatigado que lo habitual", "Me fatigo o me canso más fácilmente que lo habitual", "Estoy demasiado fatigado o cansado para hacer muchas de las cosas que solía hacer", "Estoy demasiado fatigado o cansado para hacer la mayoría de las cosas que solía hacer"]),
        ("21. Pérdida de Interés en el Sexo", ["No he notado ningún cambio reciente en mi interés por el sexo", "Estoy menos interesado en el sexo de lo que solía estarlo", "Estoy mucho menos interesado en el sexo", "He perdido completamente el interés en el sexo"])
    ]
    
    bdi_respuestas = {}
    for i, (titulo, opciones) in enumerate(bdi_preguntas):
        st.write(f"**{titulo}**")
        val = st.radio(f"Selec_{i}", options=opciones, key=f"bdi_{i}", label_visibility="collapsed", index=None)
        
        if val is not None:
            bdi_respuestas[i] = opciones.index(val)
        
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Continuar", use_container_width=True):
            # Validamos si la cantidad de respuestas guardadas es menor al total de preguntas
            if len(bdi_respuestas) < len(bdi_preguntas):
                nombre_usuario = st.session_state.data.get('nombre', 'participante')
                st.error(f"Lo siento {nombre_usuario}, parece que dejaste algunas preguntas sin responder, por favor completa el formulario.")
            else:
                # Si todas están respondidas, sumamos y avanzamos a la pantalla 7
                st.session_state.data['bdi_puntaje'] = sum(bdi_respuestas.values())
                st.session_state.screen = 7
                st.rerun()

# ------------------------------------------
# PANTALLA 7: RIESGO DE SUICIDIO (ASQ)
# ------------------------------------------
elif screen == 7:
    st.markdown("> ℹ️ La siguiente escala no se utiliza directamente en el modelo de predicción, pero es muy importante para identificar posibles riesgos de suicidio.")
    st.subheader("Escala de Ideación Suicida de Columbia (C-SSRS / ASQ)")
    
    asq_1 = st.radio("1. ¿Has tenido pensamientos de que estarías mejor muerto/a o de hacerte daño?", ["Sí", "No"], key="asq_1")
    asq_2 = st.radio("2. ¿Has tenido pensamientos de hacerte daño?", ["Sí", "No"], key="asq_2")
    asq_3 = st.radio("3. ¿Has pensado acerca de cómo podrías hacerte daño?", ["Sí", "No"], key="asq_3")
    asq_4 = st.radio("4. ¿Has tenido algún plan de hacerte daño?", ["Sí", "No"], key="asq_4")
    
    tiene_riesgo = (asq_1 == "Sí" or asq_2 == "Sí" or asq_3 == "Sí" or asq_4 == "Sí")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Continuar", use_container_width=True):
            st.session_state.data['asq_puntaje'] = 1 if tiene_riesgo else 0
            
            # Asignación de probabilidades base
            st.session_state.data['p_bayes'] = 0.35 if st.session_state.data['antecedente'] == "con-diagnostico" else 0.15
            st.session_state.data['p_texto'] = 0.0  
            st.session_state.data['p_combinada'] = st.session_state.data['p_bayes']
            
            # INSERCIÓN EN SUPABASE (Ocurre siempre al finalizar las pruebas)
            if supabase:
                try:
                    # Intento de inserción en participantes_estudio
                    res_part = supabase.table("participantes_estudio").insert({
                        "edad": st.session_state.data['edad'],
                        "sexo": st.session_state.data['sexo'],
                        "escolaridad": st.session_state.data['escolaridad'],
                        "orientacion": st.session_state.data['orientacion'],
                        "antecedente": st.session_state.data['antecedente'],
                        "respuesta_abierta": st.session_state.data['pregunta_abierta'],
                        "bdi_puntaje": st.session_state.data['bdi_puntaje'],
                        "asq_puntaje": st.session_state.data['asq_puntaje'],
                        "p_bayes": st.session_state.data['p_bayes'],
                        "p_texto": st.session_state.data['p_texto'],
                        "p_combinada": st.session_state.data['p_combinada']
                    }).execute()
                    
                    st.success("¡Participante guardado con éxito en la tabla principal!") # <-- Mensaje de prueba
                    
                    if res_part.data and len(res_part.data) > 0:
                        participante_id = res_part.data[0]['id']
                        contacto_raw = st.session_state.data['contacto']
                        contacto_cifrado = cipher_suite.encrypt(contacto_raw.encode('utf-8')).decode('utf-8')
                        
                        supabase.table("contactos_cifrados").insert({
                            "participante_id": participante_id,
                            "contacto_cifrado": contacto_cifrado
                        }).execute()
                        st.success("¡Contacto cifrado guardado con éxito!") # <-- Mensaje de prueba
                except Exception as e:
                    # ESTO MOSTRARÁ EL ERROR REAL EN PANTALLA
                    st.error(f"❌ ERROR CRÍTICO EN SUPABASE: {e}")
                    st.stop() # Detiene la ejecución aquí para que puedas leer el error exacto
            else:
                st.error("El cliente de Supabase no está conectado.")

            # Direccionamiento según el nivel de riesgo o gravedad
            if tiene_riesgo:
                st.session_state.asq_triggered = True
                st.session_state.screen = 10  # Protocolo Suicidio
            elif st.session_state.data['bdi_puntaje'] >= 29:
                st.session_state.screen = 9   # Protocolo Depresión Grave
            else:
                st.session_state.screen = 8   # Resultados normales
            st.rerun()
# ------------------------------------------
# PANTALLA 8: RESULTADOS Y RECOMENDACIONES
# ------------------------------------------
elif screen == 8:
    st.subheader("Tus resultados")
    nombre_usuario= st.session_state.data.get('nombre', 'participante')
    st.write(f"Gracias por participar en este programa de prueba. {nombre_usuario}, recuerda que los resultados que aparecen a continuación no reflejan el resultado de una evaluación psicológica formal, recuerda acudir con un profesional si lo consideras necesario")
    # MOSTRAR PUNTAJES
    bdi_val = st.session_state.data.get('bdi_puntaje', 0)
    asq_val = st.session_state.data.get('asq_puntaje', 0)
    st.info(f"📋 **Resumen de tus evaluaciones:** BDI-II = {bdi_val} puntos | ASQ (Riesgo Suicida) = {'Detectado' if asq_val > 0 else 'No se identifica riesgo suicida'}")
    
    if bdi_val < 14:
        st.success("🟢 **Resultado de la evaluación: Riesgo bajo**\n\nNo se identifican indicios significativos de depresión en este momento.")
    else:
        st.warning("🟠 **Resultado de la evaluación: Riesgo moderado / elevado**\n\nSe identifican indicadores que sugieren la presencia de malestar emocional.")
        
    st.markdown("### Recomendaciones generales (basadas en mhGAP)")
    st.markdown("""
    * Mantén una rutina de sueño regular.
    * Realiza actividad física de forma constante.
    * Busca apoyo en tu red de confianza.
    * Considera técnicas de manejo del estrés.
    """)
    if st.button("Finalizar y salir"):
        st.session_state.clear()
        st.rerun()

# ------------------------------------------
# PANTALLA 9: PROTOCOLO DE DERIVACIÓN (DEPRESIÓN SEVERA)
# ------------------------------------------
elif screen == 9:
    st.error("⚠️ **Protocolo de derivación (síntomas graves de depresión)**")
    st.write("Está es una herramienta computacional y no constituye un diagnóstico o valoración psicológica ni psiquiátrica formal, no pretende sustituir el criterio clínico formal")
    nombre_usuario= st.session_state.data.get('nombre', 'participante')
    st.markdown(f"""
    {nombre_usuario} tu bienestar es lo más importante. A continuación, te proporcionamos teléfonos donde se te puede proporcionar atención personal, confidencial y profesional:  
    * Línea de emergencias: Llamar al **911** con servicio las 24 horas. 
	* LOCATEL: Marca al *0311 o al **55 5658 1111** servicio 24 horas.
	* Línea de la Vida: **800 911 2000**. Atención gratuita y confidencial en salud mental, disponible las 24 horas, todos los días. 
	* Atención Psicológica Call Center: Llamar al **55 5025 0855** en un horario de 8 a 18 hrs.
	* Consejo Ciudadano CDMX: Llamar al **55 5533 5533** servicio 24 horas.
	* SAPTEL: Llamar al **55 5259 8121** servicio 24 horas.
	* LOCATEL CDMX, Línea de Prevención del Suicidio: *0311 o **55 5658 1111**. Atención telefónica y por chat las 24 horas.
    """)
# MOSTRAR PUNTAJES AQUí TAMBIÉN
    bdi_val = st.session_state.data.get('bdi_puntaje', 0)
    asq_val = st.session_state.data.get('asq_puntaje', 0)
    st.warning(f"📋 **Tus puntajes registrados:** BDI-II = {bdi_val} puntos (Nivel Severo) | ASQ = {'Detectado' if asq_val > 0 else 'No se identifica riesgo suicida'}")
    if st.button("Finalizar evaluación"):
        st.session_state.clear()
        st.rerun()

# ------------------------------------------
# PANTALLA 10: PROTOCOLO DE DERIVACIÓN (RIESGO SUICIDA)
# ------------------------------------------
elif screen == 10:
    st.error("🚨 **Protocolo de derivación (riesgo de suicidio)**")
    nombre_usuario= st.session_state.data.get('nombre', 'participante')
    st.markdown(f"""
    **{nombre_usuario} si tienes pensamientos de hacerte daño o de quitarte la vida, por favor busca ayuda inmediata. No estás solo.**
    * Línea de emergencias: Llamar al **911** con servicio las 24 horas. 
    * LOCATEL: Marca al *0311 o al **55 5658 1111** servicio 24 horas.
    * Línea de la Vida: **800 911 2000**. Atención gratuita y confidencial en salud mental, disponible las 24 horas, todos los días. 
    * Atención Psicológica Call Center: Llamar al **55 5025 0855** en un horario de 8 a 18 hrs.
    * Consejo Ciudadano CDMX: Llamar al **55 5533 5533** servicio 24 horas.
    * SAPTEL: Llamar al **55 5259 8121** servicio 24 horas.
    * LOCATEL CDMX, Línea de Prevención del Suicidio: *0311 o **55 5658 1111**. Atención telefónica y por chat las 24 horas.
    """)
    # MOSTRAR PUNTAJES AQUí TAMBIÉN
    bdi_val = st.session_state.data.get('bdi_puntaje', 0)
    asq_val = st.session_state.data.get('asq_puntaje', 0)
    st.warning(f"📋 **Tus puntajes registrados:** BDI-II = {bdi_val} puntos | ASQ = {'Las respuestas proporcionada muestran indicadores de riesgo suicida ' if asq_val > 0 else 'No se identifica riesgo suicida'}")
    if st.button("Finalizar y salir"):
        st.session_state.clear()
        st.rerun()