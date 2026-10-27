"""Checks scripts/judge_agreement.py: the sample, the files, and the agreement statistics."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("judge_agreement", ROOT / "scripts" / "judge_agreement.py")
ja = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ja)


def row(n, outcome, grade=4):
    return {"call_id": f"c-{n}", "call_reason": "check_invoice", "persona": "calm", "customer_id": "0001-AAAAA",
            "outcome": outcome, "is_complaint": False, "retention_offer": None,
            "messages": [{"role": "client", "content": "Bonjour, ma facture ?"},
                         {"role": "advisor", "content": "Elle est de 49,98 € <b>."}],
            "scores": {c: {"score": grade, "explanation": "x"} for c in ja.CRITERIA}}


ROWS = [row(i, o) for i, o in enumerate(["resolved"] * 6 + ["abandoned"] * 3 + ["refused"] * 2)]


def test_sample_covers_every_outcome_and_is_stable():
    chosen = ja.pick(ROWS, 5)
    assert len(chosen) == 5 and {r["outcome"] for r in chosen} == {"resolved", "abandoned", "refused"}
    assert [r["call_id"] for r in chosen] == [r["call_id"] for r in ja.pick(ROWS, 5)]
    assert len(ja.pick(ROWS, 50)) == len(ROWS)


def test_export_hides_the_ai_grades_and_escapes_the_text(tmp_path):
    csv_path, html_path = ja.export(ROWS[:3], tmp_path)
    page = html_path.read_text(encoding="utf-8")
    assert "Bonjour, ma facture ?" in page and "&lt;b&gt;" in page          # the text is escaped
    assert "explanation" not in page and "4/5" not in page                   # no AI grade shown
    header = csv_path.read_text(encoding="utf-8-sig").splitlines()[0]
    assert header == "call_id;reason;mood;relevance;accuracy;compliance;empathy;comment"


def test_filled_grades_are_read_with_either_separator(tmp_path):
    for sep in (";", ","):
        path = tmp_path / f"grades{sep == ','}.csv"
        path.write_text(f"call_id{sep}reason{sep}mood{sep}relevance{sep}accuracy{sep}compliance{sep}empathy{sep}comment\n"
                        f"c-1{sep}x{sep}calm{sep}4{sep}5{sep}3{sep}4{sep}ok\nc-2{sep}x{sep}calm{sep}{sep}{sep}{sep}{sep}\n",
                        encoding="utf-8-sig")
        assert ja.read_grades(path) == {"c-1": {"relevance": 4, "accuracy": 5, "compliance": 3, "empathy": 4}}


def test_kappa_on_known_cases():
    assert ja.weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    assert ja.weighted_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) < 0
    assert 0 < ja.weighted_kappa([1, 2, 3, 4, 5, 3], [2, 2, 3, 5, 4, 3]) < 1
    assert ja.weighted_kappa([3], [3]) is None


def test_agreement_numbers_and_report():
    rows = [row(1, "resolved", 4), row(2, "resolved", 5), row(3, "abandoned", 2)]
    human = {"c-1": {c: 4 for c in ja.CRITERIA}, "c-2": {c: 3 for c in ja.CRITERIA}, "c-3": {c: 2 for c in ja.CRITERIA}}
    result = ja.agreement(human, rows)
    r = result["accuracy"]
    assert r["calls"] == 3 and r["exact"] == pytest.approx(0.67) and r["within_1"] == pytest.approx(0.67)
    assert r["mean_difference"] == pytest.approx(0.67)                       # the AI gave 5 where the person gave 3
    text = ja.report(result)
    assert "| accuracy | 3 | 67% | 67% | +0.67 |" in text
    assert "agree with a teammate?" in ja.report(result, "a teammate")
