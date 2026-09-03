"""QRC shared-core chunking (app/chunking.py — the byte-identical module the bench also runs). No
torch/GPU/network: bm25s is the only heavy dep and it's pure-python + already a control-plane dependency.
Dense ranking is exercised OFF (embedder=None), the same lexical-only path conftest pins for retrieval.

We may test chunking.py but must not EDIT it — a genuine bug found here is reported, not patched."""
from app import chunking

# --- span determinism + coverage -----------------------------------------------------------------

def test_spans_tile_the_text_with_no_gaps_or_overlap():
    """Spans must partition the text: contiguous (each starts where the last ended), start at 0, end
    at len(text) — so reassembling the chunks reproduces the document exactly."""
    text = ("The quarterly report covers revenue, churn and headcount across three regions. " * 12).strip()
    spans = chunking.chunk_spans(text)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    for (a1, b1), (a2, _b2) in zip(spans, spans[1:], strict=False):
        assert b1 == a2            # contiguous — no gap, no overlap
        assert a1 < b1             # non-empty
    # Concatenating the chunks is lossless (spans cover every char).
    assert "".join(text[a:b] for a, b in spans) == text


def test_spans_are_deterministic():
    """Same text -> identical spans every call (the onboard-time sidecar and query-time routing must
    agree on chunk ordinals, so drift here would misalign descs)."""
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 20
    assert chunking.chunk_spans(text) == chunking.chunk_spans(text)


def test_empty_and_tiny_text():
    assert chunking.chunk_spans("") == []
    # A short text is a single span covering all of it.
    assert chunking.chunk_spans("hi there") == [(0, len("hi there"))]


def test_runt_tail_merges_into_predecessor():
    """A tiny final chunk (< 25% of a window) merges back, so no chunk is too small to rank — but the
    spans still cover the whole text end-to-end."""
    text = "word " * 300  # long enough to force several windows plus a small tail
    spans = chunking.chunk_spans(text)
    window = chunking.CHUNK_TOKENS * chunking.CHARS_PER_TOKEN
    if len(spans) >= 2:
        assert (spans[-1][1] - spans[-1][0]) >= window // 4  # no runt survived
    assert spans[-1][1] == len(text)


# --- budget is respected -------------------------------------------------------------------------

def test_route_chunks_respects_budget():
    """A doc far over budget contributes only ~budget_tokens of chunks (measured in chars), NOT its
    whole body — that is the whole point of routing vs. handing the full doc over."""
    # A doc where exactly one region matches the query, the rest is filler, so routing must SELECT.
    filler = "irrelevant filler about gardening and weather patterns and cooking. " * 40
    needle = "The Vela observatory was commissioned in 2203 by Doctor Sasha Pol. " * 3
    text = filler + needle + filler
    docs = [{"doc_id": "d1", "text": text, "descs": None}]
    budget = 64
    routed = chunking.route_chunks("who commissioned the Vela observatory", docs,
                                   embedder=None, budget_tokens=budget)
    assert len(routed) == 1
    selected = routed[0]["text"]
    assert len(selected) < len(text)                      # it routed, didn't pass the whole doc
    # The selection stops one chunk after crossing the budget, so cap at budget + one chunk of slack.
    slack = chunking.CHUNK_TOKENS * chunking.CHARS_PER_TOKEN
    assert len(selected) <= budget * chunking.CHARS_PER_TOKEN + slack
    assert "Vela observatory" in selected                 # and it picked the answer-bearing region


def test_small_doc_passes_through_whole():
    """A doc at/under budget IS its own selection (full text), no elision."""
    docs = [{"doc_id": "d1", "text": "a short note about the vela observatory", "descs": None}]
    routed = chunking.route_chunks("vela", docs, embedder=None, budget_tokens=256)
    assert routed[0]["text"] == docs[0]["text"]


def test_compose_context_orders_and_joins_by_doc():
    """compose_context joins the routed docs' selections in INPUT order with the blank-line doc
    separator, skipping empties."""
    routed = [{"doc_id": "a", "text": "AAA"}, {"doc_id": "b", "text": ""}, {"doc_id": "c", "text": "CCC"}]
    ctx = chunking.compose_context(routed)
    assert ctx == "AAA\n\nCCC"                            # b skipped (empty), a before c (order kept)


# --- descs fold ONLY into index texts, never the served selection --------------------------------

def test_descs_influence_selection_but_not_served_text():
    """A chunk description is retrieval metadata folded into the INDEX text only — it steers which
    chunk is selected but is NEVER part of the returned (served) text."""
    # Two equal-size chunk regions; the query matches ONLY the second region's DESCRIPTION, not its body.
    region_a = "xxxx yyyy zzzz wwww " * 12
    region_b = "qqqq rrrr ssss tttt " * 12
    text = region_a + region_b
    spans = chunking.chunk_spans(text)
    n = len(spans)
    # Describe the LAST chunk with the query terms; leave the rest blank.
    descs = [""] * n
    descs[-1] = "quarterly revenue figures for the finance team"
    docs = [{"doc_id": "d1", "text": text, "descs": descs}]
    routed = chunking.route_chunks("quarterly revenue figures", docs, embedder=None, budget_tokens=48)
    served = routed[0]["text"]
    # The description terms steered selection to the last chunk region...
    assert (n - 1) in routed[0]["chunk_indices"]
    # ...but the description text itself is NOT in the served output (only the document's own words).
    assert "quarterly revenue figures" not in served
    assert "revenue" not in served


# --- parse_chunk_descs: happy + garbled ----------------------------------------------------------

def test_parse_chunk_descs_happy_path():
    reply = "1. Intro and scope.\n2. Revenue by region.\n3. Headcount and hiring plan."
    assert chunking.parse_chunk_descs(reply, 3) == [
        "Intro and scope.", "Revenue by region.", "Headcount and hiring plan.",
    ]


def test_parse_chunk_descs_tolerates_separators_and_extra_prose():
    """Different numbering separators (., ), :, -) parse; leading prose lines are ignored."""
    reply = ("Here are the chunk descriptions:\n"
             "1) First part.\n"
             "2: Second part.\n"
             "3 - Third part.\n"
             "Thanks!")
    assert chunking.parse_chunk_descs(reply, 3) == ["First part.", "Second part.", "Third part."]


def test_parse_chunk_descs_garbled_leaves_blanks():
    """Missing / out-of-range / too-short lines leave '' (routing then falls back to the chunk text)."""
    reply = "1. Only the first is usable.\n2.\ngarbage line with no number\n9. out of range"
    out = chunking.parse_chunk_descs(reply, 3)
    assert out[0] == "Only the first is usable."
    assert out[1] == "" and out[2] == ""                  # blank/garbled/out-of-range -> ''
    assert len(out) == 3                                  # always length n_chunks


def test_parse_chunk_descs_empty_reply():
    assert chunking.parse_chunk_descs("", 4) == ["", "", "", ""]
