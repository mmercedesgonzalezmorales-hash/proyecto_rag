"""
app.py — Interfaz Streamlit de MicroTutor (bonus del proyecto).

Ejecutar en local:   streamlit run app.py
Desplegar:           Streamlit Community Cloud (ver README).

Streamlit solo pinta: el agente (grafo LangGraph + ChromaDB) vive en rag_core.py.
"""
# Streamlit Cloud trae un sqlite3 antiguo para ChromaDB; si pysqlite3 está instalado, lo usamos.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import uuid

import streamlit as st

st.set_page_config(page_title="MicroTutor · Microbiología con IA", page_icon="🦠", layout="centered")

# En Streamlit Cloud la API key se configura en "Secrets"; en local, en el archivo .env
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

import rag_core  # noqa: E402  (después de configurar la API key)


@st.cache_resource(show_spinner="Preparando la base de conocimiento (solo la primera vez)…")
def cargar_agente(api_key: str):
    """Se cachea por API key: si el usuario cambia la clave, se crea un agente nuevo."""
    os.environ["GOOGLE_API_KEY"] = api_key
    vectorstore = rag_core.build_vectorstore()
    return rag_core.build_agent(vectorstore), vectorstore._collection.count()


# --------------------------------------------------------------------------- estado de sesión
# Streamlit re-ejecuta el script entero en cada interacción: lo que deba perdurar va en session_state.
# Cada sesión tiene su propio thread_id -> su propia memoria en el checkpointer de LangGraph.
st.session_state.setdefault("thread_id", str(uuid.uuid4()))
st.session_state.setdefault("historial", [])

# --------------------------------------------------------------------------- barra lateral
with st.sidebar:
    st.title("🦠 MicroTutor")
    st.caption("Tutor de **microbiología** construido con RAG sobre el libro "
               "*Microbiology* de OpenStax (6 capítulos). Recurso de estudio: no sustituye a un "
               "profesional sanitario.")
    if not os.getenv("GOOGLE_API_KEY"):
        clave = st.text_input("Google API Key", type="password",
                              help="Consíguela gratis en https://aistudio.google.com/apikey")
        if clave:
            os.environ["GOOGLE_API_KEY"] = clave
    mostrar_traza = st.toggle("Mostrar ruta y fuentes", value=True)
    if st.button("🧹 Nueva conversación", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.historial = []
        st.rerun()
    st.divider()
    st.markdown("**Prueba a preguntar:**\n"
                "- ¿Qué diferencia hay entre gram positivas y gram negativas?\n"
                "- ¿Cuáles son las fases de la curva de crecimiento?\n"
                "- ¿Cómo se vuelven resistentes las bacterias a los antibióticos?")
    st.caption(f"Modelo: `{rag_core.LLM_MODEL}` · Embeddings: `{rag_core.EMBEDDING_MODEL}`")

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("Falta la GOOGLE_API_KEY. Introdúcela en la barra lateral.")
    st.stop()

try:
    agent, n_chunks = cargar_agente(api_key)
except Exception as exc:
    st.error(f"No se pudo iniciar el agente: {exc}")
    st.stop()

st.sidebar.caption(f"📚 {n_chunks} fragmentos indexados en ChromaDB")


def pintar_traza(msg: dict):
    if msg.get("ruta") == "buscar":
        with st.expander(f"📚 Consultó el libro · {len(msg['sources'])} fragmentos · "
                         f"búsqueda: “{msg['search_query']}”"):
            st.caption(f"Motivo: {msg.get('motivo', '')}")
            for s in msg["sources"]:
                st.markdown(f"**[{s['n']}]** `{s['source']}` › {s['section']} — relevancia {s['score']}")
                st.caption(s["preview"] + "…")
    elif msg.get("ruta") == "directo":
        st.caption(f"💬 Respuesta directa, sin consultar el libro — {msg.get('motivo', '')}")


st.title("Pregúntale al libro de microbiología")
if not st.session_state.historial:
    st.info("Hola, soy MicroTutor. Pregúntame sobre la célula microbiana, virus, crecimiento y control "
            "de microorganismos, antibióticos o patogenicidad. Recuerdo la conversación, así que puedes "
            "hacer preguntas de seguimiento.")

for msg in st.session_state.historial:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and mostrar_traza:
            pintar_traza(msg)

if pregunta := st.chat_input("Escribe tu pregunta…"):
    st.session_state.historial.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando…"):
            try:
                r = rag_core.ask(agent, pregunta, thread_id=st.session_state.thread_id)
            except Exception as exc:
                st.error(f"Error al consultar a Gemini: {exc}")
                st.stop()
        msg = {"role": "assistant", "content": r["answer"], "ruta": r["ruta"], "motivo": r["motivo"],
               "sources": r["sources"], "search_query": r["search_query"]}
        st.markdown(r["answer"])
        if mostrar_traza:
            pintar_traza(msg)

    st.session_state.historial.append(msg)
