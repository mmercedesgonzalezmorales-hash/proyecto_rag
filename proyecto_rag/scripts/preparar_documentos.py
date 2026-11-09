"""Regenera la carpeta data/ a partir del libro de texto abierto *Microbiology* (OpenStax).

Convierte los módulos CNXML (XML) del repositorio oficial en 3 Markdown limpios, uno por
bloque temático, conservando títulos de capítulo, sección y subsección. Elimina figuras,
imágenes, ejercicios, preguntas de repaso, enlaces y cajas interactivas (ruido para el RAG).

Uso (no hace falta para ejecutar el proyecto: data/ ya viene generado):
    git clone --depth 1 https://github.com/openstax/osbooks-microbiology /tmp/osmicro
    python scripts/preparar_documentos.py /tmp/osmicro data

Fuente: OpenStax, *Microbiology*. Parker, Schneegurt, Thi Tu, Forster y Lister.
Rice University, 2016. https://openstax.org/details/books/microbiology
Licencia: CC BY-NC-SA 4.0 (uso educativo no comercial, con atribución).
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

# Clases de bloques que no aportan contenido explicativo (preguntas, enlaces, casos interactivos...)
SKIP_CLASSES = {"check-your-understanding", "media-2", "interactive-exercise", "summary",
                "review-questions", "multiple-choice", "fill-in-the-blank", "short-answer",
                "critical-thinking", "learning-objectives", "true-false", "matching",
                "link-to-learning", "clinical-focus", "case-in-point"}
SKIP_TAGS = {"figure", "media", "image", "exercise", "metadata", "link-to-learning"}


def local(tag: str) -> str:
    return tag.split("}", 1)[-1]


def inline_text(el) -> str:
    """Texto de un elemento con sus hijos en línea (term, emphasis, sub, sup...)."""
    parts = [el.text or ""]
    for child in el:
        name = local(child.tag)
        if name in SKIP_TAGS:
            pass
        elif name in {"sub", "sup"}:
            parts.append(inline_text(child))
        elif name == "term":
            parts.append(f"**{inline_text(child)}**")
        elif name == "newline":
            parts.append(" ")
        else:
            parts.append(inline_text(child))
        parts.append(child.tail or "")
    text = re.sub(r"\s+", " ", "".join(parts))
    text = re.sub(r"\s*\((\s|,|and|;)*\)", "", text)   # referencias a figuras eliminadas: "()" o "(, )"
    return text.strip()


def render(el, depth: int, out: list[str]):
    for child in el:
        name = local(child.tag)
        cls = set((child.get("class") or "").split())
        if name in SKIP_TAGS or cls & SKIP_CLASSES:
            continue
        if name == "section":
            title = child.find("{*}title")
            if title is not None:
                t = inline_text(title)
                if t.lower() in {"key terms", "summary", "review questions", "critical thinking",
                                 "fill in the blank", "short answer", "multiple choice"}:
                    continue
                out.append(f"\n{'#' * min(depth, 6)} {t}\n")
            render(child, depth + 1, out)
        elif name == "para":
            text = inline_text(child)
            if text:
                out.append(text + "\n\n")
            # listas anidadas dentro del párrafo
            for lst in child.findall("{*}list"):
                render_list(lst, out)
        elif name == "list":
            render_list(child, out)
        elif name == "note":
            title = child.find("{*}title")
            label = inline_text(title) if title is not None else "Nota"
            out.append(f"\n> **{label}**\n")
            sub: list[str] = []
            render(child, depth + 1, sub)
            out.extend("> " + line if line.strip() else line for line in "".join(sub).splitlines(True))
        elif name == "table":
            render_table(child, out)
        elif name in {"title", "caption", "label"}:
            continue
        else:
            render(child, depth, out)


def render_list(lst, out):
    for item in lst.findall("{*}item"):
        text = inline_text(item)
        if text:
            out.append(f"- {text}\n")
    out.append("\n")


def render_table(table, out):
    title = table.find("{*}title")
    if title is not None:
        out.append(f"\n**{inline_text(title)}**\n\n")
    rows = []
    for row in table.iter():
        if local(row.tag) == "row":
            rows.append([inline_text(e).replace("|", "/") for e in row if local(e.tag) == "entry"])
    if not rows:
        return
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out.append("| " + " | ".join(rows[0]) + " |\n")
    out.append("|" + "---|" * width + "\n")
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |\n")
    out.append("\n")


def module_to_md(module_id: str) -> tuple[str, str]:
    root = ET.parse(REPO / "modules" / module_id / "index.cnxml").getroot()
    title = inline_text(root.find("{*}title"))
    out: list[str] = []
    render(root.find("{*}content"), 3, out)
    body = re.sub(r"\n{3,}", "\n\n", "".join(out)).strip()
    return title, body


def chapters() -> dict[str, list[str]]:
    """Capítulo -> lista de módulos, leído del índice del libro."""
    xml = (REPO / "collections" / "microbiology.collection.xml").read_text()
    result, current = {}, None
    for m in re.finditer(r'<col:subcollection>\s*<md:title>(.*?)</md:title>|<col:module document="(m\d+)"', xml):
        if m.group(1):
            current = m.group(1)
            result[current] = []
        elif current:
            result[current].append(m.group(2))
    return result


# Bloques temáticos de la base de conocimiento: (archivo, título, [(capítulo, nº de capítulo)])
DOCS = [
    ("01_celula_y_virus.md", "La célula microbiana y los patógenos acelulares",
     [("The Cell", 3), ("Acellular Pathogens", 6)]),
    ("02_crecimiento_y_control.md", "Crecimiento microbiano y su control",
     [("Microbial Growth", 9), ("Control of Microbial Growth", 13)]),
    ("03_antimicrobianos_y_patogenicidad.md", "Fármacos antimicrobianos y mecanismos de patogenicidad",
     [("Antimicrobial Drugs", 14), ("Microbial Mechanisms of Pathogenicity", 15)]),
]

CREDIT = ("Fuente: OpenStax, *Microbiology* (Parker, Schneegurt, Thi Tu, Forster y Lister; "
          "Rice University, 2016), https://openstax.org/details/books/microbiology. "
          "Licencia CC BY-NC-SA 4.0. Extracto limpiado (sin figuras ni ejercicios) para uso educativo.")

index = chapters()
for fname, doc_title, chapter_list in DOCS:
    parts = [f"# {doc_title}\n\n{CREDIT}\n"]
    for chapter, number in chapter_list:
        for i, mod in enumerate(index[chapter]):
            title, body = module_to_md(mod)
            if title == "Introduction":
                title = f"Capítulo {number}: {chapter} — introducción"
            else:
                title = f"{number}.{i} {title}"
            parts.append(f"\n\n# {title}\n\n{body}")
    text = "".join(parts)
    (OUT / fname).write_text(text, encoding="utf-8")
    words = len(text.split())
    print(f"{fname}: {words} palabras (~{words // 500} páginas)")
