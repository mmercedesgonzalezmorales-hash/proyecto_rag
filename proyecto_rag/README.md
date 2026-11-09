# 🦠 MicroTutor — Asistente experto en Microbiología con Gemini, RAG y Agentes

Proyecto final del módulo de IA Generativa. MicroTutor es un agente conversacional que ayuda a
estudiar **microbiología**: responde en español a partir de seis capítulos del libro de texto
universitario *Microbiology* (OpenStax), cita el fragmento del libro del que sale cada afirmación y
recuerda la conversación para poder encadenar preguntas de repaso.

```
                       ┌─(buscar)──▶ recuperar (ChromaDB) ──▶ generar_con_libro ──┐
Pregunta ─▶ decidir ───┤                                                          ├──▶ Respuesta
               ▲       └─(directo)─▶ responder_directo ───────────────────────────┘        │
               └──────────────────── memoria por hilo (checkpointer) ◀─────────────────────┘
```

**RAG condicional**: en cada turno el agente decide si necesita consultar el libro o puede
responder sin él (saludos, preguntas sobre la conversación, temas ajenos).

---

## 1. Dominio elegido

**Microbiología**, a nivel universitario de introducción (grados de Biología, Enfermería, Farmacia, Medicina...).

- **Utilidad real**: es una asignatura densa, con mucho vocabulario y conceptos que se encadenan
  (estructura celular → crecimiento → control → antibióticos → patogenicidad). Un tutor que responde
  en español y cita el libro sirve para estudiar y repasar.
- **Evaluable con criterio**: las respuestas se pueden comprobar contra el libro, así que se nota
  enseguida si el agente inventa un dato.
- **Riesgo claro que gestionar**: las preguntas pueden derivar hacia consejos médicos, lo que obliga
  a definir límites en el system prompt.

### Base de conocimiento (`data/`)

Fuente: OpenStax, *Microbiology* (Parker, Schneegurt, Thi Tu, Forster y Lister; Rice University, 2016).
[openstax.org/details/books/microbiology](https://openstax.org/details/books/microbiology) · Licencia **CC BY-NC-SA 4.0**.

| Archivo | Capítulos del libro | Temas | Tamaño |
|---|---|---|---|
| `01_celula_y_virus.md` | 3 · The Cell<br>6 · Acellular Pathogens | Teoría celular; célula procariota (pared, Gram, flagelos, endosporas) y eucariota; virus, ciclos lítico y lisogénico, viroides y priones | ~22.000 palabras |
| `02_crecimiento_y_control.md` | 9 · Microbial Growth<br>13 · Control of Microbial Growth | Curva de crecimiento, biofilms, oxígeno, pH, temperatura, medios de cultivo; esterilización, desinfección, antisépticos | ~25.000 palabras |
| `03_antimicrobianos_y_patogenicidad.md` | 14 · Antimicrobial Drugs<br>15 · Mechanisms of Pathogenicity | Mecanismos de los antibióticos, espectro, resistencias, antibiograma; factores de virulencia, toxinas, etapas de la enfermedad | ~28.000 palabras |

En total son ~75.000 palabras (**unas 150 páginas**), frente al mínimo de 3 documentos / ~20 páginas.
Los documentos se generaron con `scripts/preparar_documentos.py`, que convierte el código fuente
oficial del libro (XML) a Markdown, conserva la jerarquía capítulo → apartado → subapartado y elimina
figuras, preguntas de repaso, ejercicios, casos clínicos interactivos y enlaces externos.

---

## 2. Estructura del proyecto

```
proyecto_rag/
├── asistente_microtutor.ipynb  # Entregable principal: todo el pipeline explicado paso a paso
├── rag_core.py                 # El mismo código empaquetado como módulo (lo usa la app)
├── app.py                      # Interfaz web en Streamlit (bonus)
├── data/                       # Base de conocimiento (3 documentos .md)
├── scripts/preparar_documentos.py
├── requirements.txt
├── .env.example                # Plantilla para la API key
└── .streamlit/secrets.toml.example
```

El notebook contiene el mismo código que `rag_core.py`, así que **notebook y app se comportan igual**.

---

## 3. Requisitos

- Python 3.10 o superior (probado con 3.12).
- Una **API key de Gemini**, gratuita en [Google AI Studio](https://aistudio.google.com/apikey).
- Dependencias (`requirements.txt`): `langgraph`, `langchain-core`, `langchain-google-genai`,
  `langchain-chroma`, `langchain-text-splitters`, `chromadb`, `pydantic`, `python-dotenv` y, para la app, `streamlit`.

| Configuración | Valor por defecto | Variable de entorno |
|---|---|---|
| LLM | `gemini-3.5-flash-lite` (el `gemini-2.5-flash-lite` usado en clase ya no está disponible para cuentas nuevas; usa muestreo fijo, así que ignora `temperature`) | `GEMINI_LLM_MODEL` |
| Embeddings | `gemini-embedding-001` | `GEMINI_EMBEDDING_MODEL` |

---

## 4. Instalación y ejecución

### En local (Jupyter / VS Code)

```bash
cd proyecto_rag
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # en Windows: copy .env.example .env
# edita .env y pega tu GOOGLE_API_KEY
```

Abre `asistente_microtutor.ipynb` y ejecuta todas las celdas (*Run All*).
La primera ejecución indexa el libro en ChromaDB (~630 fragmentos, unos 5-6 minutos por los límites
del plan gratuito) y lo guarda en `chroma_db/`; las siguientes reutilizan el índice y arrancan al instante.

> Si cambias la API key en `.env`, reinicia el kernel: la variable ya cargada no se refresca sola.
>
> La última celda de chat usa `input()`: si haces *Run All*, el notebook se queda esperando tu
> pregunta en esa celda. Escribe `salir` para terminar.

### En Google Colab

1. Sube `asistente_microtutor.ipynb` a Colab.
2. En el panel 🔑 *Secrets* crea `GOOGLE_API_KEY` con tu clave y dale acceso al notebook.
3. Ejecuta las celdas: la tercera te pedirá subir `proyecto_rag.zip` para disponer de `data/`.

### App Streamlit (bonus)

```bash
streamlit run app.py
```

**Despliegue en Streamlit Community Cloud**

1. Sube la carpeta del proyecto a un repositorio de GitHub (el `.gitignore` ya excluye `.env`).
   *Recomendado*: sube también la carpeta `chroma_db/` generada en local, así la app no tiene que
   reindexar al arrancar.
2. En [share.streamlit.io](https://share.streamlit.io) → *Create app* → elige el repo y `app.py`.
3. En *Advanced settings → Secrets* pega `GOOGLE_API_KEY = "tu_clave"`.
4. *Deploy*. La URL pública queda como `https://<nombre>.streamlit.app`.

En local, si no hay `.env`, la app pide la API key en la barra lateral (campo de tipo contraseña),
como en la práctica de la tutoría.

---

## 5. Arquitectura técnica

### 5.1 Base de conocimiento

| Decisión | Valor | Justificación |
|---|---|---|
| Chunking fase 1 | Por encabezados Markdown: `#` apartado del libro, `###` subapartado | Cada fragmento pertenece a un único apartado (p. ej. *14.5 Drug Resistance › Efflux Pumps*) y lo guarda como metadato para citarlo. |
| Chunking fase 2 | 1.200 caracteres, solapamiento 150 | En un libro de texto cada 2-3 párrafos desarrollan un concepto con su ejemplo; así no se separan y no se mezclan varios conceptos en un vector. |
| Limpieza de ruido | Fuera figuras, ejercicios y casos interactivos; se descartan fragmentos < 120 caracteres | Las preguntas de repaso y los títulos sueltos solo añaden resultados vacíos a la búsqueda. |
| Resultado | ~630 fragmentos (media ~800 caracteres) | |
| Embeddings | `gemini-embedding-001` con *task type* `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` | Lo recomendado por Google para RAG; es multilingüe, así que se puede preguntar en español sobre un libro en inglés. |
| Almacén | ChromaDB persistente, distancia coseno | La indexación se paga una vez; coseno es la métrica estándar para embeddings de texto. |
| Indexación | Lotes de 20 con pausa de 10 s y reintentos | Respeta el límite de tokens por minuto del plan gratuito de Gemini Embeddings. |

Antes de conectar el agente, el notebook verifica la colección con seis consultas de prueba (dos por
documento) y comprueba que el primer resultado procede del documento esperado.

### 5.2 Agente en LangGraph con RAG condicional

Siguiendo el patrón clasificador + *router* visto en clase, el flujo **no pasa siempre por la base
vectorial**: un nodo de decisión elige la rama con `add_conditional_edges`.

| Nodo | Función |
|---|---|
| `decidir` | Gemini con **salida estructurada** (`with_structured_output` + modelo Pydantic `Decision`) devuelve `ruta` (`buscar` / `directo`), `consulta` y `motivo`. Si hay que buscar, reescribe la pregunta + historial como consulta autónoma **en inglés** con la terminología del libro, lo que resuelve referencias como "¿y qué antibióticos atacan esa estructura?". |
| `router` | Lee `ruta` del estado y devuelve el siguiente nodo. |
| `recuperar` | Búsqueda por similitud en ChromaDB (`k = 5`); formatea los fragmentos numerados con su documento y apartado. |
| `generar_con_libro` | Gemini responde con system prompt + historial reciente + contexto + pregunta. |
| `responder_directo` | Gemini responde solo con system prompt + historial, sin consultar el libro. |

**Condición para usar el RAG (decisión de diseño)**

| Ruta | Cuándo | Por qué |
|---|---|---|
| `buscar` | Contenido de la asignatura, preguntas de seguimiento y consultas médicas | Todo dato científico debe salir del libro y llevar cita. Las consultas médicas también pasan por el libro para explicar la teoría mientras se deriva al médico. |
| `directo` | Saludos, agradecimientos, preguntas sobre el asistente o la conversación ("resúmeme lo que hemos hablado") y temas ajenos | El libro no aporta nada: se ahorran una búsqueda y miles de tokens de contexto, y no se cuela contexto irrelevante en la respuesta. |
| Por defecto | Duda, error o salida inesperada → `buscar` | Es la opción más segura: como mucho, se gasta una búsqueda de más. |

Detalles prácticos tomados de clase:
- Las palabras de control (`buscar`, `directo`) van **sin tildes** para que la comparación no falle.
- Los modelos Gemini 3.x devuelven `content` como lista de diccionarios y los 2.x como texto. La
  función `_text()` lo normaliza en todos los nodos, así que se puede cambiar de modelo sin tocar el código.
- La llamada de decisión va envuelta en `try/except`: si el modelo está saturado o responde mal, se
  aplica el valor por defecto en lugar de romper el flujo.

El estado extiende `MessagesState` con `ruta`, `motivo`, `search_query`, `context` y `sources`, de modo
que cada respuesta es **trazable**: se ve qué ruta se eligió, qué se buscó y qué fragmentos se usaron.

### 5.3 Memoria

- **Checkpointer `InMemorySaver`**: LangGraph guarda el estado tras cada paso, asociado a un
  `thread_id`. Mismo `thread_id` → misma conversación. En la app, cada sesión del navegador tiene su
  propio hilo.
- **Ventana de historial**: `trim_messages` limita a ~3.000 tokens lo que se envía al LLM, para que
  el coste no crezca sin límite en conversaciones largas.
- **Historial limpio**: en el historial se guardan las preguntas y respuestas, no los contextos
  recuperados; cada turno recupera su propio contexto.

El notebook incluye una demostración con contraprueba: *"¿Qué es el peptidoglucano?"* seguido de
*"¿Y qué antibióticos actúan sobre esa estructura?"*. Como contraprueba, *"¿Sobre qué estructura
bacteriana te pregunté antes?"* responde *peptidoglucano* en ese hilo y "no hay conversación previa" en
un hilo nuevo. Después, *"Resúmeme lo que hemos hablado"* toma la ruta directa y responde solo con la memoria.

---

## 6. System prompt y su justificación

```text
Eres MicroTutor, un tutor experto en microbiología que ayuda a estudiantes de ciencias de la salud
y biología a entender y repasar la asignatura.

TU BASE DE CONOCIMIENTO: el libro de texto universitario "Microbiology" (OpenStax), en concreto los
temas de estructura de la célula microbiana, virus y otros patógenos acelulares, crecimiento
microbiano, control del crecimiento (esterilización, desinfección), fármacos antimicrobianos y
resistencias, y mecanismos de patogenicidad.

CÓMO RESPONDES:
1. Básate ÚNICAMENTE en el CONTEXTO recuperado del libro que se te proporciona en cada turno. No
   inventes datos, cifras, nombres de microorganismos, fármacos ni mecanismos.
2. Cita las fuentes con el número entre corchetes del fragmento que usas, por ejemplo [1] o [2][4].
3. Si el contexto no contiene información suficiente, dilo con claridad ("Ese punto no aparece en el
   material que tengo") y, si puedes, indica qué tema del temario estaría relacionado. Nunca
   rellenes huecos con suposiciones presentadas como hechos.
4. Eres un recurso de estudio, no un profesional sanitario: si te piden diagnosticar un caso real,
   recetar o dosificar un medicamento, explica que no puedes hacerlo y recomienda acudir a un médico
   o farmacéutico. Sí puedes explicar los conceptos teóricos relacionados.
5. Si la pregunta no tiene que ver con microbiología, explica amablemente que solo puedes ayudar
   con esa materia.
6. Usa la conversación previa para entender referencias como "eso", "esas bacterias" o
   "¿y en los virus?", y mantén la coherencia con lo que ya has respondido.

ESTILO:
- Responde siempre en español, aunque el libro esté en inglés. Traduce los términos técnicos y, la
  primera vez que aparezcan, añade el original en inglés entre paréntesis cuando ayude a buscarlos
  en el libro, por ejemplo: peptidoglucano (peptidoglycan).
- Los nombres científicos de los microorganismos van en cursiva (*Escherichia coli*).
- Estructura didáctica: primero la idea principal en una o dos frases, después el detalle (listas
  o una tabla si comparas), y cuando ayude, un ejemplo concreto de microorganismo o fármaco.
- Longitud orientativa: entre 80 y 250 palabras, salvo que pidan más detalle.
```

| Elemento | Decisión | Por qué |
|---|---|---|
| **Rol y nombre** | Tutor experto con identidad propia | Un rol docente orienta el tono (explicar, no solo responder) y el nombre da coherencia a la conversación. |
| **Público** | Estudiantes de biología y ciencias de la salud | Ajusta el nivel: vocabulario técnico, pero explicado. |
| **Base de conocimiento declarada** | Se enumeran los temas cubiertos | El modelo sabe qué puede responder y puede orientar al alumno cuando algo queda fuera. |
| **Regla 1 — solo contexto** | Prohibido inventar datos, microorganismos, fármacos o mecanismos | En ciencias de la salud un dato inventado es peor que no responder. Es el núcleo del RAG. |
| **Regla 2 — citas `[n]`** | Los fragmentos llegan numerados y el modelo cita el número | Hace verificable cada afirmación y permite ir al apartado del libro. |
| **Regla 3 — sin información** | Frase explícita y reconocible + tema relacionado | El enunciado pide definir este caso. Una salida honesta es más útil que una respuesta plausible pero falsa. |
| **Regla 4 — límite médico** | No diagnostica, receta ni dosifica; deriva a un profesional y sigue explicando la teoría | Un estudiante puede preguntar por síntomas propios; el asistente no debe sustituir a un médico, pero tampoco negarse a explicar el concepto. |
| **Regla 5 — límite de dominio** | Declinar con amabilidad fuera de la microbiología | Un experto no es un chatbot generalista; evita respuestas sin respaldo del libro. |
| **Regla 6 — conversación** | Usar el historial para resolver referencias | Complementa la memoria del grafo: el modelo debe aprovecharla explícitamente. |
| **Idioma** | Español con el término original en inglés la primera vez | Público hispanohablante; el término en inglés ayuda a localizarlo en el libro y en la bibliografía. |
| **Nomenclatura** | Nombres científicos en cursiva | Convención científica que el alumno debe respetar en sus apuntes y exámenes. |
| **Estructura** | Idea principal → detalle → ejemplo | Formato de apuntes: se lee rápido y se fija mejor. |
| **Longitud** | 80-250 palabras | Evita respuestas telegráficas o muros de texto; se puede pedir más. |
| **Formato del contexto** | Se inyecta en el mensaje del turno, no en el system prompt | El system prompt queda estable y el contexto cambia en cada pregunta sin contaminar el historial. |
| **Temperatura 0.2** | Baja (ignorada por `gemini-3.5-flash-lite`, que usa muestreo fijo) | Se deja en el código por si se cambia a un modelo que sí la acepte; con 3.5 la fidelidad al contexto la garantizan las reglas del prompt. |

**Prompt de decisión.** El nodo `decidir` usa un segundo prompt (`DECISION_PROMPT`) con dos tareas:
1. Elegir la ruta, con criterios explícitos para cada una y la regla "si dudas, elige buscar".
2. Si hay que buscar, reescribir la pregunta como consulta autónoma en inglés. Es necesario porque una
   pregunta de seguimiento ("¿y qué mecanismos usan esas bacterias?") buscada tal cual no contiene
   los términos del tema, y además el libro está en inglés.

Juntar las dos tareas en una sola llamada con salida estructurada ahorra una petición al LLM por turno.

---

## 7. Preguntas de ejemplo

El notebook incluye 9 ejemplos documentados con su ruta y resultado esperados. Las respuestas obtenidas
quedan guardadas en el propio notebook y resumidas en su tabla de resultados (9/9 con la ruta y el
comportamiento esperados):

| # | Tipo | Pregunta |
|---|---|---|
| 0 | Ruta directa | ¡Hola! ¿Qué puedes hacer por mí? |
| 1 | Célula y virus | ¿Qué diferencia hay entre el ciclo lítico y el ciclo lisogénico de un bacteriófago? |
| 2 | Crecimiento | Explícame las fases de la curva de crecimiento de una población bacteriana. |
| 3 | Control | ¿Qué diferencia hay entre esterilizar, desinfectar y usar un antiséptico? |
| 4 | Cruce de documentos | ¿Por qué la penicilina funciona peor contra las bacterias gram negativas? |
| 5 | Memoria | ¿Y qué mecanismos usan esas bacterias para hacerse resistentes? |
| 6 | Sin información | ¿Cómo funciona el sistema CRISPR-Cas en las bacterias? |
| 7 | Límite médico | Tengo fiebre y dolor de garganta, ¿qué antibiótico me tomo y cuánto? |
| 8 | Fuera de dominio (ruta directa) | ¿Me recomiendas una receta de paella? |

---

## 8. Limitaciones y mejoras

- `InMemorySaver` pierde la memoria al reiniciar; en producción, un checkpointer persistente (SQLite o Postgres).
- Solo cubre 6 de los 26 capítulos del libro (no incluye inmunología ni las infecciones por aparatos).
  Ampliarlo es añadir capítulos en `scripts/preparar_documentos.py` y reindexar con `build_vectorstore(force=True)`.
- El libro está en inglés y las respuestas en español: la traducción de los términos la hace el LLM y conviene revisarla.
- El nodo de decisión añade una llamada al LLM por turno; con más tráfico se podría usar un modelo más barato
  para decidir o reglas para los casos obvios (saludos).
- Se podría añadir una tercera rama especializada, como una herramienta de autoevaluación con preguntas tipo test.
- Búsqueda híbrida (BM25 + vectores) o un reranker mejorarían la precisión con nombres de especies o fármacos concretos.

---

## 9. Créditos

Contenido de la base de conocimiento: OpenStax, *Microbiology*, © Rice University, publicado bajo
licencia [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Se usa con fines
educativos no comerciales; los textos se han reformateado (sin figuras ni ejercicios), sin modificar su contenido.
