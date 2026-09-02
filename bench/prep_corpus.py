"""Build the head-to-head benchmark corpus + question set from the 13 QASPER sample papers.

Emits two files the harness (headtohead.py) consumes:
  bench/corpus.jsonl   — one {"doc_id","text"} per line; doc_id = the .txt file stem
                         (e.g. "1611_01576"). This is exactly the shape POST /onboard_cag
                         expects in its `docs` list, and the retriever indexes the same text.
  bench/questions.json — [{"q","expect_substring","doc_id"}] fact questions with EXPECTED
                         substrings, so the harness's spot-check measures a REAL answer, never
                         confabulation speed (bench_fleet's 4-request fact guard, ported).

Provenance of the questions (WHY the template, not auto-extracted):
  The established fact-grid question set lives at
    Engram-Smart-CAG/cartridges/serve/grid_questions.json
  ("source: fact-grid 2026-08-03 (grid_bf16.json), unchanged for cross-model comparability").
  Its docs are a DIFFERENT set of QASPER papers (qasper-1503-00841, ...): of the 13 sample
  papers here, only ONE overlaps — qasper-1611-01576 == 1611_01576 — with a gold-verified
  question ("Up to how many times faster are QRNNs than LSTMs..." -> "16", which occurs
  verbatim in 1611_01576.txt: "up to 16 times faster at train and test time"). We copy that
  one slot VERBATIM (real provenance) and emit EMPTY template slots for the other 12 docs.
  Programmatically inventing factual questions is explicitly NOT acceptable (the operator must
  hand-write them after reading each paper), so unfilled slots stay blank and the script prints
  a LOUD note listing exactly which docs still need questions.

Run:  python bench/prep_corpus.py
No GPU, no network — pure text I/O over the sample .txt files.
"""
from __future__ import annotations

import json
from pathlib import Path

# The 13 QASPER sample papers. Kept as an explicit path (not an env) because the corpus that
# seeds this benchmark is fixed — the same 13 papers the fact-grid evidence was built against.
QASPER_DIR = Path(r"c:\Users\willg\Github\Engram-Smart-CAG\qasper_samples")
# The established, cross-model-comparable question set (gold answers verified against the docs).
GRID_QUESTIONS = Path(
    r"c:\Users\willg\Github\Engram-Smart-CAG\cartridges\serve\grid_questions.json")

BENCH_DIR = Path(__file__).resolve().parent
CORPUS_OUT = BENCH_DIR / "corpus.jsonl"
QUESTIONS_OUT = BENCH_DIR / "questions.json"


def _grid_doc_id_to_stem(grid_doc_id: str) -> str:
    """Map a grid doc_id to a sample file stem: 'qasper-1611-01576' -> '1611_01576'.
    The grid namespaces with 'qasper-' and joins the arXiv id parts with '-'; the sample files
    drop the namespace and join with '_' (the file stem). This is the ONLY id shape difference
    between the two corpora, so it is the whole mapping."""
    body = grid_doc_id[len("qasper-"):] if grid_doc_id.startswith("qasper-") else grid_doc_id
    return body.replace("-", "_")


def _load_grid_questions() -> dict[str, list[dict]]:
    """{stem: [{"q","expect_substring","doc_id"}]} for every grid question whose doc maps to a
    sample stem AND whose gold answer actually occurs in that doc's text (so a copied slot is a
    genuine, verifiable spot-check). Empty when the grid file is absent."""
    if not GRID_QUESTIONS.exists():
        return {}
    grid = json.loads(GRID_QUESTIONS.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for q in grid.get("questions", []):
        stem = _grid_doc_id_to_stem(q["doc_id"])
        txt = QASPER_DIR / f"{stem}.txt"
        if not txt.exists():
            continue  # grid doc not among the 13 samples
        gold = str(q.get("gold", "")).strip()
        # Only copy the slot if the gold string literally appears in the doc — the spot-check is a
        # substring test, so a gold that isn't present verbatim can't serve as expect_substring.
        if not gold or gold not in txt.read_text(encoding="utf-8"):
            continue
        out.setdefault(stem, []).append(
            {"q": q["question"], "expect_substring": gold, "doc_id": stem})
    return out


def build_corpus() -> list[str]:
    """Write corpus.jsonl from the 13 .txt papers; return the ordered doc_id (stem) list.
    Text is the full file verbatim — the first line is the paper title (the serve stack keys its
    'According to ...' sourcing on the first line, and onboarding truncates to CAG_MAX_DOC_TOK)."""
    files = sorted(QASPER_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"no QASPER .txt files under {QASPER_DIR}")
    stems: list[str] = []
    with CORPUS_OUT.open("w", encoding="utf-8") as fh:
        for f in files:
            stem = f.stem
            text = f.read_text(encoding="utf-8")
            fh.write(json.dumps({"doc_id": stem, "text": text}, ensure_ascii=False) + "\n")
            stems.append(stem)
    print(f"[prep] wrote {len(stems)} docs -> {CORPUS_OUT}")
    return stems


def build_questions(stems: list[str]) -> None:
    """Write questions.json: real gold-verified slots where the grid overlaps a sample doc, empty
    template slots (blank q/expect_substring) for every other doc. Loud note lists the blanks."""
    grid = _load_grid_questions()
    questions: list[dict] = []
    filled_stems: list[str] = []
    template_stems: list[str] = []
    for stem in stems:
        if stem in grid:
            questions.extend(grid[stem])
            filled_stems.append(stem)
        else:
            # TEMPLATE slot — the operator hand-writes q + expect_substring after reading the doc.
            # Emit TWO empty slots per doc to match the "2-3 factual questions per doc" target.
            for _ in range(2):
                questions.append({"q": "", "expect_substring": "", "doc_id": stem})
            template_stems.append(stem)
    QUESTIONS_OUT.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[prep] wrote {len(questions)} question slots -> {QUESTIONS_OUT}")
    if filled_stems:
        print(f"[prep] PRE-FILLED (verbatim from grid_questions.json, gold verified in-doc): "
              f"{', '.join(filled_stems)}")
    if template_stems:
        # LOUD note: these slots are BLANK on purpose. Auto-generating factual questions is not
        # acceptable; the operator must fill q + expect_substring by hand before running accuracy.
        print("\n" + "=" * 78)
        print("!! ACTION REQUIRED: hand-write questions for the docs below (BLANK template slots).")
        print("!! Read each paper, add 2-3 clearly-answerable factual questions per doc, and set")
        print("!! expect_substring to a short string that MUST appear in a correct answer.")
        print(f"!! Docs still needing questions ({len(template_stems)} of {len(stems)}):")
        for s in template_stems:
            print(f"!!    - {s}")
        print("=" * 78 + "\n")


def main() -> None:
    stems = build_corpus()
    build_questions(stems)


if __name__ == "__main__":
    main()
