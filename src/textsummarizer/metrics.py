import re

from rouge_score import rouge_scorer

ROUGE_KEYS = ("rouge1", "rouge2", "rougeL", "rougeLsum")

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Split on newlines and on whitespace after `.`, `!` or `?`; no nltk data needed."""
    return [piece.strip() for piece in _SENTENCE_BREAK.split(text) if piece and piece.strip()]


def prepare_for_rouge(text: str) -> str:
    """Turn Pegasus's `<n>` marker into a newline, then put one sentence on each line.

    ROUGE-Lsum scores line by line, so both predictions and references need this.
    """
    return "\n".join(split_sentences(text.replace("<n>", "\n")))


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Mean ROUGE F-measure times 100 (rounded to 2 decimals) for each of ROUGE_KEYS."""
    if not predictions or len(predictions) != len(references):
        raise ValueError(
            f"need the same, non-zero number of predictions and references, "
            f"got {len(predictions)} and {len(references)}"
        )
    scorer = rouge_scorer.RougeScorer(list(ROUGE_KEYS), use_stemmer=True)
    totals = dict.fromkeys(ROUGE_KEYS, 0.0)
    for prediction, reference in zip(predictions, references, strict=True):
        scores = scorer.score(prepare_for_rouge(reference), prepare_for_rouge(prediction))
        for key in ROUGE_KEYS:
            totals[key] += scores[key].fmeasure
    return {key: round(100 * totals[key] / len(predictions), 2) for key in ROUGE_KEYS}


def bertscore_f1(predictions: list[str], references: list[str], model_type: str) -> float:
    """Mean BERTScore F1 times 100. Downloads `model_type` on first use."""
    import torch
    from bert_score import score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, f1 = score(
        predictions, references, model_type=model_type, lang="en", device=device, verbose=False
    )
    return round(100 * float(f1.mean()), 2)
