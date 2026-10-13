from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from llm_router_serving.errors import ModelContractError
from llm_router_serving.inference import TransformersNliReranker
from llm_router_serving.sentence_diff import reduce_pair

# ── sentence_diff ──


def test_shared_sentences_are_dropped_and_the_changed_one_kept():
    passage = "The library has many rooms. It opened in 1990. "
    a, b = reduce_pair(passage + "Who founded it?", passage + "When did it close?")
    assert (a, b) == ("Who founded it?", "When did it close?")


def test_numbers_are_never_folded_into_a_shared_sentence():
    a, b = reduce_pair(
        "We use Python 3.11.  The list has 120 items.",
        "we use python 3.11. The list has 150 items.",
    )
    # Case and spacing differ in the first sentence only, so it is shared;
    # the second differs in a digit and must survive.
    assert (a, b) == ("The list has 120 items.", "The list has 150 items.")


def test_nothing_shared_returns_the_originals():
    a, b = reduce_pair("How do I sort a list?", "How do I reverse an array?")
    assert (a, b) == ("How do I sort a list?", "How do I reverse an array?")


def test_pure_insertion_keeps_context_so_neither_side_is_blank():
    a, b = reduce_pair(
        "Add two numbers. Return the sum.",
        "Add two numbers. Return the sum. Use C++.",
    )
    assert a and b
    assert b.endswith("Use C++.")


def test_identical_texts_are_returned_unchanged():
    text = "Explain recursion. Give an example."
    assert reduce_pair(text, text) == (text, text)


# ── TransformersNliReranker ──


class PairTokenizer:
    """Records the directed pairs and encodes row i as input id i."""

    def __init__(self):
        self.pairs = []

    def __call__(self, firsts, seconds, **kwargs):
        assert kwargs["truncation"] is True
        self.pairs = list(zip(firsts, seconds, strict=True))
        return {"input_ids": torch.arange(len(firsts)).unsqueeze(1)}


class TableModel:
    """Returns the logits stored for each directed pair."""

    def __init__(self, tokenizer, table):
        self._tokenizer = tokenizer
        self._table = table

    def __call__(self, **inputs):
        rows = [self._table[pair] for pair in self._tokenizer.pairs]
        return SimpleNamespace(logits=torch.tensor(rows, dtype=torch.float32))


def _engine(table, entailment_index=0):
    tokenizer = PairTokenizer()
    return tokenizer, TransformersNliReranker(
        tokenizer=tokenizer,
        model=TableModel(tokenizer, table),
        torch=torch,
        device=torch.device("cpu"),
        model_version="tasksource/deberta-base-long-nli@04dcf11",
        max_length=1280,
        entailment_index=entailment_index,
    )


def _entail(logits, index=0):
    return float(torch.softmax(torch.tensor(logits), 0)[index])


def test_score_is_the_lower_entailment_of_the_two_directions():
    q, c = "Find the largest prime.", "Find the smallest prime."
    table = {(q, c): [3.0, 0.0, 0.0], (c, q): [-2.0, 0.0, 1.0]}
    tokenizer, engine = _engine(table)

    result = engine.rerank(q, [c])

    assert tokenizer.pairs == [(q, c), (c, q)]   # both directions in one forward pass
    assert result.scores == pytest.approx([min(_entail(table[(q, c)]), _entail(table[(c, q)]))])
    assert result.model_version == "tasksource/deberta-base-long-nli@04dcf11"


def test_scores_stay_aligned_with_candidates():
    q = "How do I sort a list in Python?"
    c1, c2 = "How do I sort a list in Java?", "What is the way to sort a Python list?"
    table = {
        (q, c1): [-3.0, 0.0, 3.0], (c1, q): [-3.0, 0.0, 3.0],
        (q, c2): [4.0, 0.0, 0.0], (c2, q): [4.0, 0.0, 0.0],
    }
    _, engine = _engine(table)

    scores = engine.rerank(q, [c1, c2]).scores

    assert scores[1] > 0.9 > 0.1 > scores[0]


def test_the_verifier_reads_only_the_changed_sentence():
    passage = "Kristy baked cookies for her friends. She ate two of them. "
    q, c = passage + "How many were left?", passage + "How many did she bake?"
    table = {
        ("How many were left?", "How many did she bake?"): [0.0, 1.0, 0.0],
        ("How many did she bake?", "How many were left?"): [0.0, 1.0, 0.0],
    }
    tokenizer, engine = _engine(table)

    engine.rerank(q, [c])

    assert tokenizer.pairs[0] == ("How many were left?", "How many did she bake?")


def test_entailment_column_is_found_by_label_name():
    config = SimpleNamespace(id2label={0: "contradiction", 1: "ENTAILMENT", 2: "neutral"})
    assert TransformersNliReranker._entailment_index(config) == 1


@pytest.mark.parametrize(
    "id2label",
    [
        {0: "LABEL_0"},
        {0: "contradiction", 1: "neutral"},
        {0: "entailment", 1: "entailment"},
        None,
    ],
)
def test_checkpoint_without_one_entailment_label_is_refused(id2label):
    with pytest.raises(ModelContractError):
        TransformersNliReranker._entailment_index(SimpleNamespace(id2label=id2label))


@pytest.mark.parametrize("nan_direction", [0, 1])
def test_non_finite_score_fails_loudly_in_either_direction(nan_direction):
    q, c = "a", "b"
    rows = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    rows[nan_direction] = [float("nan"), 0.0, 0.0]
    table = {(q, c): rows[0], (c, q): rows[1]}
    _, engine = _engine(table)
    with pytest.raises(ModelContractError):
        engine.rerank(q, [c])
