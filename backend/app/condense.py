"""Conversational query condensation (UPGRADE 1). A follow-up turn's raw text ("what about its
pricing?") routes and retrieves badly because the retrieval stack (retrieve / _hybrid_split) sees
only the pronoun-laden question, not the topic it refers to. standalone_question() augments that
follow-up with the topic terms it implicitly refers to, so the SAME retrieval stack that turn 1
used sees a resolved query. The augmented form is used ONLY for retrieval/routing — the question
SERVED to the model stays the original (the model already has the history as prefill).

WHY DETERMINISTIC, NOT AN LLM REWRITE (2026-09-03, measured live on the box): the first
implementation asked the engine to rewrite the question (the classic condense-question chain).
Command A+ deliberates in-answer nearly unconditionally in that framing — "We need to rewrite the
user's last question... the pronoun 'it' refers to..." — across every instruction style tried
(blunt demand, one-shot example, completion-slot anchor) and every budget (48/96 tokens), either
burning the budget before the artifact appeared or costing 4-6s per follow-up turn to maybe
finish. Retrieval doesn't need grammar; bm25s and the dense embedder need the TOPIC WORDS the
pronoun hides. So: append the prior user turn's salient terms to the follow-up, deterministically,
at zero latency and zero GPU. The LLM rewrite can return for models that comply cheaply.

Graceful no-op by design: history empty, a question that is already information-dense, or nothing
salient to add all return None, and the caller uses the original question everywhere."""
import logging
import re

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]+")

# Stopwords + interrogatives + the anaphora words themselves: everything that carries no topic
# signal. Anything NOT here that appears in the prior user turn is treated as topic-salient.
_NOISE = frozenset("""
a an and are as at be but by did do does for from had has have how i in is it its itself of on or
so that the their them they this those to was were what when where which who whose why will with
you your about into over under between against during then now also just there here please
tell me us can could would should
""".split())

# A follow-up with at least this many informative tokens is treated as already standalone —
# augmenting it would only dilute the query (and the debug signal): return None instead.
_STANDALONE_TOKENS = 8

# Cap on appended topic terms: enough to anchor bm25/dense on the referent, few enough that the
# follow-up's own words still dominate the query.
_MAX_TOPIC_TERMS = 12


def _informative(text: str) -> list[str]:
    """The topic-salient tokens of `text`, order-preserving, deduped, noise removed."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in _WORD.findall(text):
        low = tok.lower()
        if low in _NOISE or low in seen:
            continue
        seen.add(low)
        out.append(tok)
    return out


def standalone_question(history: list[dict], question: str) -> str | None:
    """Augment `question` with the topic terms of the most recent USER turns, or None to signal
    "use the original everywhere" (history empty, the question is already information-dense, or
    nothing salient to add). The result is `<original question> <topic terms>` — ugly to a human,
    exactly what lexical+dense ranking wants, and it can never narrate, fail, or add latency."""
    if not history:
        return None
    own = _informative(question)
    if len(own) >= _STANDALONE_TOKENS:
        return None
    own_low = {t.lower() for t in own}
    # Walk user turns newest-first: the referent of a pronoun is almost always the most recent
    # topic. Assistant turns are skipped — they often paraphrase broadly and dilute the terms.
    terms: list[str] = []
    for msg in reversed(history):
        if msg.get("role") != "user":
            continue
        for tok in _informative(msg.get("content") or ""):
            if tok.lower() in own_low:
                continue
            terms.append(tok)
            own_low.add(tok.lower())
            if len(terms) >= _MAX_TOPIC_TERMS:
                break
        if terms:
            break  # the most recent user turn with salient terms is the referent
    if not terms:
        return None
    return f"{question} {' '.join(terms)}"
