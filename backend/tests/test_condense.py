"""Query condensation (app/condense.py) — the DETERMINISTIC topic-term augmenter.

The first implementation asked the engine for an LLM rewrite; Command A+ deliberated in-answer
across every instruction style and budget tried live on the box (2026-09-03), so condensation is
now pure string work: append the prior user turn's salient terms to a pronoun-laden follow-up.
These tests pin that contract: no engine involvement ever, graceful None on turn 1 / standalone /
nothing-salient, augmentation preserves the original question verbatim up front."""
from app import condense


def _hist(*turns):
    """[(role, content), ...] -> history dicts."""
    return [{"role": r, "content": c} for r, c in turns]


def test_empty_history_is_none():
    assert condense.standalone_question([], "what about its pricing?") is None


def test_pronoun_followup_gets_prior_user_topic_terms():
    h = _hist(("user", "What dataset does the paper on joint ontology learning use?"),
              ("assistant", "It uses Wikipedia first sentences."))
    out = condense.standalone_question(h, "And what type of grammar does it induce?")
    assert out is not None
    # The original question leads verbatim; the referent's topic terms follow.
    assert out.startswith("And what type of grammar does it induce?")
    for term in ("dataset", "paper", "joint", "ontology", "learning"):
        assert term in out
    # Assistant turns are skipped as term sources (they paraphrase broadly).
    assert "Wikipedia" not in out


def test_information_dense_question_is_left_alone():
    h = _hist(("user", "Tell me about the onboarding pipeline."),
              ("assistant", "It builds one cart per document."))
    q = ("Switching topics: in what language was the railway information "
         "speech system developed and deployed?")
    assert condense.standalone_question(h, q) is None  # >= threshold informative tokens


def test_no_salient_prior_terms_is_none():
    # Prior user turn is all noise words -> nothing to append -> None.
    h = _hist(("user", "so what about that then"), ("assistant", "Could you clarify?"))
    assert condense.standalone_question(h, "and it was?") is None


def test_duplicate_terms_not_appended():
    h = _hist(("user", "What does the grammar induction paper say about grammar?"),)
    out = condense.standalone_question(h, "What grammar does it use?")
    assert out is not None
    # 'grammar' is already in the question -> appended terms must not repeat it.
    appended = out[len("What grammar does it use?"):]
    assert "grammar" not in appended.lower()


def test_most_recent_user_turn_wins():
    h = _hist(("user", "Summarize the billing policy."),
              ("assistant", "..."),
              ("user", "Now describe the railway speech system."),
              ("assistant", "..."))
    out = condense.standalone_question(h, "What language is it in?")
    assert out is not None and "railway" in out
    assert "billing" not in out  # older topic must not bleed in


def test_never_touches_the_engine(monkeypatch):
    """The deterministic augmenter must have NO ml_client dependency — a network/GPU outage can
    never break condensation, and no per-turn latency is added."""
    import app.ml_client as ml_client

    def _boom(*a, **k):  # noqa: ANN001, ANN002
        raise AssertionError("condense must not call the engine")

    for name in ("inference_rag", "inference_query"):
        if hasattr(ml_client, name):
            monkeypatch.setattr(ml_client, name, _boom)
    h = _hist(("user", "What dataset does the ontology paper use?"),)
    assert condense.standalone_question(h, "And what does it induce?") is not None
