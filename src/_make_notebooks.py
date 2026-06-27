import nbformat as nbf
import re
import os

STAGE_FILES = [
    ("01_case_base.py", "Tahap 1: Membangun Case Base"),
    ("02_case_representation.py", "Tahap 2: Case Representation"),
    ("03_retrieval.py", "Tahap 3: Case Retrieval"),
    ("04_predict.py", "Tahap 4: Case/Solution Reuse"),
    ("05_evaluation.py", "Tahap 5: Model Evaluation"),
]

SRC_DIR = os.path.dirname(__file__)
NB_DIR = os.path.join(SRC_DIR, "..", "notebooks")
os.makedirs(NB_DIR, exist_ok=True)


def split_into_cells(code: str):
    """Pisahkan kode menjadi beberapa cell logis: docstring module -> markdown,
    import block -> 1 cell, lalu setiap top-level def/class -> cell sendiri,
    blok `if __name__` -> cell eksekusi terakhir."""
    lines = code.split("\n")

    # Ambil module docstring (di antara baris pertama yang \"\"\" hingga \"\"\" penutup)
    docstring = ""
    body_start = 0
    if lines and lines[0].startswith("#!"):
        body_start = 1
    j = body_start
    while j < len(lines) and not lines[j].strip().startswith('"""'):
        j += 1
    if j < len(lines):
        k = j + 1
        while k < len(lines) and '"""' not in lines[k]:
            k += 1
        docstring = "\n".join(lines[j:k+1])
        body_start = k + 1

    rest = "\n".join(lines[body_start:]).strip("\n")

    # split top-level blocks: import section, then each top-level def, then main-call
    blocks = []
    current = []
    for line in rest.split("\n"):
        if re.match(r"^(def |if __name__)", line) and current:
            blocks.append("\n".join(current).strip("\n"))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip("\n"))

    return docstring, [b for b in blocks if b.strip()]


def main():
    for fname, title in STAGE_FILES:
        py_path = os.path.join(SRC_DIR, fname)
        with open(py_path, "r", encoding="utf-8") as f:
            code = f.read()

        docstring, blocks = split_into_cells(code)
        nb = nbf.v4.new_notebook()
        cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n```\n{docstring.strip(chr(34)*3)}\n```")]
        for b in blocks:
            cells.append(nbf.v4.new_code_cell(b))
        nb["cells"] = cells

        nb_path = os.path.join(NB_DIR, fname.replace(".py", ".ipynb"))
        with open(nb_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"[OK] {nb_path}")


if __name__ == "__main__":
    main()
