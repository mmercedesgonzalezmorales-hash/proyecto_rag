"""
rag_core.py — Núcleo del asistente MicroTutor (RAG + LangGraph + Gemini).

Lo usan la app de Streamlit (app.py) y cualquier script que quiera reutilizar
el agente. El notebook contiene este mismo código explicado paso a paso.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable, Literal

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Configuración (se puede sobrescribir con variables de entorno / .env)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "microbiologia"

LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

CHUNK_SIZE = 1200      # caracteres por fragmento
CHUNK_OVERLAP = 150    # solapamiento para no cortar ideas a la mitad
TOP_K = 5              # fragmentos recuperados por pregunta
MAX_HISTORY_TOKENS = 3000  # historial que se envía al LLM (memoria de ventana)

SYSTEM_PROMPT = """Eres MicroTutor, un tutor experto en microbiología que ayuda a estudiantes \
de ciencias de la salud y biología a entender y repasar la asignatura.

TU BASE DE CONOCIMIENTO: el libro de texto universitario "Microbiology" (OpenStax), en concreto \
los temas de estructura de la célula microbiana, virus y otros patógenos acelulares, crecimiento \
microbiano, control del crecimiento (esterilización, desinfección), fármacos antimicrobianos y \
resistencias, y mecanismos de patogenicidad.

CÓMO RESPONDES:
1. Básate ÚNICAMENTE en el CONTEXTO recuperado del libro que se te proporciona en cada turno. \
No inventes datos, cifras, nombres de microorganismos, fármacos ni mecanismos.
2. Cita las fuentes con el número entre corchetes del fragmento que usas, por ejemplo [1] o [2][4].
3. Si el contexto no contiene información suficiente, dilo con claridad ("Ese punto no aparece \
en el material que tengo") y, si puedes, indica qué tema del temario estaría relacionado. Nunca \
rellenes huecos con suposiciones presentadas como hechos.
4. Eres un recurso de estudio, no un profesional sanitario: si te piden diagnosticar un caso real, \
recetar o dosificar un medicamento, explica que no puedes hacerlo y recomienda acudir a un médico \
o farmacéutico. Sí puedes explicar los conceptos teóricos relacionados.
5. Si la pregunta no tiene que ver con microbiología, explica amablemente que solo puedes ayudar \
con esa materia.
6. Usa la conversación previa para entender referencias como "eso", "esas bacterias" o \
"¿y en los virus?", y mantén la coherencia con lo que ya has respondido.

ESTILO:
- Responde siempre en español, aunque el libro esté en inglés. Traduce los términos técnicos \
y, la primera vez que aparezcan, añade el original en inglés entre paréntesis cuando ayude \
a buscarlos en el libro, por ejemplo: peptidoglucano (peptidoglycan).
- Los nombres científicos de los microorganismos van en cursiva (*Escherichia coli*).
- Estructura didáctica: primero la idea principal en una o dos frases, después el detalle \
(listas o una tabla si comparas), y cuando ayude, un ejemplo concreto de microorganismo o fármaco.
- Longitud orientativa: entre 80 y 250 palabras, salvo que pidan más detalle."""

DECISION_PROMPT = """Eres el nodo de enrutamiento de MicroTutor, un tutor de microbiología. \
Decide si para responder la ÚLTIMA pregunta del usuario hay que consultar el libro de texto \
de microbiología (base vectorial).

Elige "buscar" si la pregunta necesita contenido de la asignatura: conceptos, microorganismos, \
virus, estructuras celulares, crecimiento, esterilización, antibióticos, resistencias, toxinas, \
enfermedades... Incluye las preguntas de seguimiento ("¿y en los virus?", "¿por qué pasa eso?") \
y las que piden consejo médico (para poder explicar la teoría relacionada).

Elige "directo" si NO hace falta el libro: saludos, despedidas, agradecimientos, preguntas \
sobre el propio asistente, peticiones sobre la conversación ya mantenida ("resúmeme lo que \
hemos hablado", "¿qué te pregunté antes?") y temas sin relación con la microbiología.

Si dudas, elige "buscar".

Si eliges "buscar", escribe en "consulta" una búsqueda autónoma para un buscador semántico sobre \
un libro en inglés: resuelve las referencias a la conversación previa y usa la terminología \
científica en inglés que usaría el libro. Si eliges "directo", deja "consulta" vacía.
En "motivo" justifica la decisión en una frase.

Conversación previa:
{history}

Última pregunta: {question}"""


def require_api_key() -> str:
    """Comprueba que la API key de Gemini está en el entorno (nunca en el código)."""
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "Falta la API key de Gemini. Crea un archivo .env con GOOGLE_API_KEY=... "
            "(consíguela gratis en https://aistudio.google.com/apikey)."
        )
    os.environ["GOOGLE_API_KEY"] = key
    return key


def get_embeddings() -> Embeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def get_llm(temperature: float = 0.2) -> BaseChatModel:
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=temperature)


# ---------------------------------------------------------------------------
# 1. Base de conocimiento: carga, limpieza, chunking e indexación en ChromaDB
# ---------------------------------------------------------------------------
def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Lee los .md de la carpeta data/ como Documents con su archivo de origen."""
    docs = []
    for path in sorted(data_dir.glob("*.md")):
        docs.append(Document(page_content=path.read_text(encoding="utf-8"),
                             metadata={"source": path.name}))
    if not docs:
        raise FileNotFoundError(f"No hay documentos .md en {data_dir}")
    return docs


def split_documents(docs: Iterable[Document],
                    chunk_size: int = CHUNK_SIZE,
                    chunk_overlap: int = CHUNK_OVERLAP) -> list[Document]:
    """Chunking en dos fases: primero por encabezados Markdown (respeta la estructura
    del libro y guarda el apartado como metadato) y después por tamaño."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "section"), ("###", "subsection")],
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n#### ", "\n> **", "\n\n", "\n", ". ", " "],
    )
    chunks: list[Document] = []
    for doc in docs:
        for part in header_splitter.split_text(doc.page_content):
            part.metadata = {**doc.metadata, **part.metadata}
            chunks.extend(size_splitter.split_documents([part]))
    # Limpieza de ruido: descartar fragmentos casi vacíos (solo un título, una línea de código...)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 120]
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def build_vectorstore(chunks: list[Document] | None = None,
                      embeddings: Embeddings | None = None,
                      persist_dir: Path | str | None = CHROMA_DIR,
                      batch_size: int = 20,
                      pause_s: float = 10.0,
                      force: bool = False) -> Chroma:
    """Crea (o reabre) la colección de ChromaDB.

    Si la colección ya existe en disco y no se pide `force`, se reutiliza: así no se
    vuelven a gastar llamadas de embeddings cada vez que se arranca el notebook o la app.
    La indexación va por lotes con una pausa para no superar el límite de tokens por minuto
    del plan gratuito de Gemini Embeddings.
    """
    embeddings = embeddings or get_embeddings()
    vs = Chroma(collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(persist_dir) if persist_dir else None,
                collection_metadata={"hnsw:space": "cosine"})
    if force:
        vs.reset_collection()
    if vs._collection.count() > 0:
        return vs

    chunks = chunks if chunks is not None else split_documents(load_documents())
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        for attempt in range(5):
            try:
                vs.add_documents(batch, ids=[f"chunk-{c.metadata['chunk_id']}" for c in batch])
                break
            except Exception as exc:  # típico: 429 por límite de peticiones
                wait = 30 * (attempt + 1)
                print(f"  Lote {start}: {type(exc).__name__}; reintento en {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"No se pudo indexar el lote que empieza en {start}")
        if pause_s:
            time.sleep(pause_s)
    return vs


def format_context(results: list[tuple[Document, float]]) -> str:
    blocks = []
    for i, (doc, score) in enumerate(results, start=1):
        where = doc.metadata.get("source", "?")
        section = doc.metadata.get("subsection") or doc.metadata.get("section") or ""
        blocks.append(f"[{i}] (fuente: {where} — {section}; relevancia {score:.2f})\n"
                      f"{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 2. Agente LangGraph con RAG condicional y memoria por hilo
#
#    START -> decidir --(buscar)--> recuperar -> generar_con_libro -> END
#                     \-(directo)--> responder_directo ---------------> END
# ---------------------------------------------------------------------------
class Decision(BaseModel):
    """Salida estructurada del nodo `decidir`."""
    ruta: Literal["buscar", "directo"] = Field(
        description="buscar si hace falta consultar el libro; directo si no")
    consulta: str = Field(
        default="", description="Si ruta=buscar: consulta de búsqueda autónoma en inglés")
    motivo: str = Field(default="", description="Justificación breve de la decisión")


class AgentState(MessagesState):
    """Estado del grafo. `messages` (heredado, con el reducer add_messages) acumula la
    conversación; el resto de campos se sobrescribe en cada turno."""
    ruta: str
    motivo: str
    search_query: str
    context: str
    sources: list[dict]


def _history_as_text(messages, max_turns: int = 6) -> str:
    lines = []
    for m in messages[-max_turns * 2:]:
        role = "Usuario" if isinstance(m, HumanMessage) else "Asistente"
        lines.append(f"{role}: {_text(m)[:600]}")
    return "\n".join(lines) or "(sin conversación previa)"


def _text(msg) -> str:
    """Los modelos Gemini 3.x devuelven `content` como lista de diccionarios y los 2.x como
    string (la trampa vista en clase). Esta función lo deja siempre en texto."""
    content = msg.content
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


def build_agent(vectorstore: Chroma,
                llm: BaseChatModel | None = None,
                checkpointer=None,
                top_k: int = TOP_K,
                system_prompt: str = SYSTEM_PROMPT):
    llm = llm or get_llm()
    decisor = llm.with_structured_output(Decision)
    checkpointer = checkpointer or InMemorySaver()

    def _recent_history(state: AgentState):
        """Memoria de ventana: últimos ~MAX_HISTORY_TOKENS tokens, sin la pregunta actual."""
        return trim_messages(state["messages"][:-1], max_tokens=MAX_HISTORY_TOKENS,
                             token_counter="approximate", strategy="last",
                             start_on="human", include_system=False)

    def decidir(state: AgentState) -> dict:
        """Nodo clasificador: decide si hay que consultar la base vectorial y, en ese caso,
        reescribe la pregunta como consulta autónoma en inglés."""
        question = _text(state["messages"][-1])
        prompt = DECISION_PROMPT.format(history=_history_as_text(state["messages"][:-1]),
                                        question=question)
        try:
            d = decisor.invoke(prompt)
            ruta, consulta, motivo = d.ruta, d.consulta.strip(), d.motivo
        except Exception as exc:  # respuesta mal formada o API saturada
            ruta, consulta, motivo = "buscar", "", f"valor por defecto ({type(exc).__name__})"
        if ruta not in ("buscar", "directo"):
            ruta = "buscar"  # ante la duda, la opción más segura es consultar el libro
        return {"ruta": ruta, "motivo": motivo, "search_query": consulta or question,
                "context": "", "sources": []}

    def router(state: AgentState) -> Literal["recuperar", "responder_directo"]:
        return "recuperar" if state["ruta"] == "buscar" else "responder_directo"

    def recuperar(state: AgentState) -> dict:
        results = vectorstore.similarity_search_with_relevance_scores(state["search_query"], k=top_k)
        sources = [{"n": i, "source": d.metadata.get("source"),
                    "section": d.metadata.get("subsection") or d.metadata.get("section"),
                    "score": round(s, 3), "preview": d.page_content[:200]}
                   for i, (d, s) in enumerate(results, start=1)]
        return {"context": format_context(results), "sources": sources}

    def generar_con_libro(state: AgentState) -> dict:
        question = _text(state["messages"][-1])
        grounded = HumanMessage(content=(
            f"CONTEXTO recuperado del libro de microbiología:\n\n{state['context']}\n\n"
            f"=====\nPREGUNTA DEL USUARIO: {question}"))
        answer = llm.invoke([SystemMessage(content=system_prompt), *_recent_history(state), grounded])
        return {"messages": [AIMessage(content=_text(answer))]}

    def responder_directo(state: AgentState) -> dict:
        question = _text(state["messages"][-1])
        direct = HumanMessage(content=(
            "(En este turno NO se ha consultado el libro: la pregunta no lo requiere. Responde "
            "con la conversación previa si hace falta. No aportes datos de microbiología nuevos "
            "sin contexto del libro.)\n\n"
            f"PREGUNTA DEL USUARIO: {question}"))
        answer = llm.invoke([SystemMessage(content=system_prompt), *_recent_history(state), direct])
        return {"messages": [AIMessage(content=_text(answer))]}

    graph = StateGraph(AgentState)
    graph.add_node("decidir", decidir)
    graph.add_node("recuperar", recuperar)
    graph.add_node("generar_con_libro", generar_con_libro)
    graph.add_node("responder_directo", responder_directo)
    graph.add_edge(START, "decidir")
    graph.add_conditional_edges("decidir", router,
                                {"recuperar": "recuperar", "responder_directo": "responder_directo"})
    graph.add_edge("recuperar", "generar_con_libro")
    graph.add_edge("generar_con_libro", END)
    graph.add_edge("responder_directo", END)
    return graph.compile(checkpointer=checkpointer)


def ask(agent, question: str, thread_id: str = "demo") -> dict:
    """Envía una pregunta al agente dentro de un hilo de conversación (memoria)."""
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.invoke({"messages": [HumanMessage(content=question)]}, config)
    return {"answer": _text(state["messages"][-1]),
            "ruta": state["ruta"],
            "motivo": state.get("motivo", ""),
            "search_query": state["search_query"],
            "sources": state["sources"]}
