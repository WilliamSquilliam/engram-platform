"""Engine char->token mapping (ml_service/vllm_inference._spans_to_token_ranges) and the RESIDENT-QRC
attribution turn (_resident_user_turn) — the two PURE functions the resident span-load path factors
out. Loaded from the vLLM service module BY FILE PATH (it lives outside the backend `app` package and
guards its heavy vllm/cartridges/fastapi imports), so these run with NO GPU, NO vLLM, NO network — the
math over a fake tokenizer's offset mapping is all that's under test.

Covers: a char span expands to FULLY cover its chars (a straddling token is included); clipping to the
truncated token length; empty/degenerate spans drop; multiple spans stay in ascending source order; and
the attribution turn lists titles in order with a direct-answer instruction (and no text otherwise)."""
import importlib.util as ilu
from pathlib import Path

import pytest

_VI_PATH = Path(__file__).resolve().parents[2] / "ml_service" / "vllm_inference.py"


def _load_engine():
    """Load vllm_inference by path (it's not importable as a package and its vllm imports are guarded
    inside functions, so module import is GPU-free)."""
    spec = ilu.spec_from_file_location("vllm_inference_under_test", _VI_PATH)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vi = _load_engine()

# A fake tokenizer offset mapping: 6 tokens over 20 chars. offsets[i] = (char_start, char_end) of tok i.
#   tok0 [0,3)  tok1 [3,4)  tok2 [4,9)  tok3 [9,10)  tok4 [10,15)  tok5 [15,20)
OFFSETS = [(0, 3), (3, 4), (4, 9), (9, 10), (10, 15), (15, 20)]


def test_span_expands_to_fully_cover_chars():
    """A char span [4,15) covers tokens 2,3,4 (each overlaps the range) -> token range [2,5)."""
    assert vi._spans_to_token_ranges(OFFSETS, [[4, 15]], 6) == [(2, 5)]


def test_span_includes_straddling_boundary_tokens():
    """A span [2,11) starts inside tok0 [0,3) and ends inside tok4 [10,15): every overlapping token is
    included (never dropped for straddling the boundary) -> [0,5)."""
    assert vi._spans_to_token_ranges(OFFSETS, [[2, 11]], 6) == [(0, 5)]


def test_range_clips_to_truncated_token_count():
    """With only n_tokens=3 available (doc truncated to CAG_MAX_DOC_TOK), span [4,15) keeps only the
    tokens < 3 that overlap -> just tok2 -> [2,3)."""
    assert vi._spans_to_token_ranges(OFFSETS, [[4, 15]], 3) == [(2, 3)]


def test_span_entirely_past_truncation_drops():
    """A span whose chars live only in tokens beyond the truncation point resolves EMPTY (dropped),
    never a zero-token segment: span [15,20) with n_tokens=4 (tok5 excluded) -> []."""
    assert vi._spans_to_token_ranges(OFFSETS, [[15, 20]], 4) == []


def test_degenerate_and_zero_width_spans_drop():
    """A zero-width [5,5) or inverted [9,4) span contributes nothing."""
    assert vi._spans_to_token_ranges(OFFSETS, [[5, 5], [9, 4]], 6) == []


def test_multiple_spans_ascending_source_order():
    """Two spans map to two token ranges in ascending source order."""
    assert vi._spans_to_token_ranges(OFFSETS, [[0, 4], [10, 20]], 6) == [(0, 2), (4, 6)]


def test_empty_special_offset_tokens_skipped():
    """A tokenizer that emits (0,0) offsets for special tokens (shouldn't with add_special_tokens=False,
    but be robust): those tokens are skipped, real tokens still map."""
    offsets = [(0, 0), (0, 4), (4, 8), (0, 0)]  # tok0/tok3 are special (empty offset)
    assert vi._spans_to_token_ranges(offsets, [[0, 8]], 4) == [(1, 3)]


def test_resident_user_turn_lists_titles_in_order():
    """The attribution turn names spanned-doc titles in order + a direct-answer instruction + the Q."""
    turn = vi._resident_user_turn("What is the reactor fuel?", ["Doc B", "Doc C"])
    assert "excerpts from: Doc B" in turn and "Doc C." in turn
    assert turn.index("Doc B") < turn.index("Doc C")
    assert "Answer directly" in turn
    assert turn.rstrip().endswith("What is the reactor fuel?")


def test_resident_user_turn_no_titles_is_bare_question():
    """No surviving spans -> a pure top-1 cart serve: the turn is just the question (no attribution)."""
    assert vi._resident_user_turn("Q?", []) == "Q?"


def test_resident_user_turn_single_title():
    """One spanned doc reads naturally ('excerpts from: <title>.') without a dangling 'and'."""
    turn = vi._resident_user_turn("Q?", ["Only Doc"])
    assert "excerpts from: Only Doc." in turn and "; and" not in turn
