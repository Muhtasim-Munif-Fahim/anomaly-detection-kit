import numpy as np
import pytest

from anomaly_detection import report as rep


class TestScoreSummary:
    def test_contains_name_and_stats(self):
        text = rep.score_summary("isolation forest", np.array([0.1, 0.2, 0.9]), 1)
        assert "### isolation forest" in text
        assert "0.9000" in text
        assert "flagged: 1 of 3" in text

    def test_empty_scores_ok(self):
        text = rep.score_summary("x", np.array([]), 0)
        assert "- no data" in text


class TestFlaggedRowsTable:
    def test_header_and_cells(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scores = np.array([0.1, 0.9, 0.5])
        flags = np.array([0, 1, 1])
        text = rep.flagged_rows_table(X, scores, flags)
        assert "row" in text and "f0" in text and "score" in text
        assert "| 1 |" in text
        assert "| 3.000 | 4.000 | 0.900 |" in text

    def test_sorted_by_score_descending(self):
        X = np.zeros((4, 2))
        scores = np.array([0.2, 0.8, 0.9, 0.1])
        flags = np.ones(4)
        text = rep.flagged_rows_table(X, scores, flags, max_rows=10)
        assert text.index("| 2 |") < text.index("| 1 |") < text.index("| 0 |")

    def test_omitted_rows_note(self):
        X = np.zeros((10, 1))
        scores = np.arange(10.0)
        flags = np.ones(10)
        text = rep.flagged_rows_table(X, scores, flags, max_rows=3)
        assert "7 more flagged rows omitted" in text

    def test_no_flags_no_rows(self):
        text = rep.flagged_rows_table(np.zeros((5, 2)), np.zeros(5), np.zeros(5))
        assert "| 0 |" not in text


class TestSweepTable:
    def test_header_and_rows(self):
        rows = [
            {"threshold": 0.1, "precision": 1.0, "recall": 0.5, "f1": 0.6667, "tp": 2, "fp": 0, "fn": 2},
            {"threshold": 0.9, "precision": 0.5, "recall": 1.0, "f1": 0.6667, "tp": 4, "fp": 4, "fn": 0},
        ]
        text = rep.sweep_table(rows)
        assert "threshold" in text and "precision" in text
        assert "0.1000" in text
        assert "| 2 | 0 | 2 |" in text

    def test_empty(self):
        assert rep.sweep_table([]) == ""


class TestComparisonTable:
    def test_header_and_rows(self):
        rows = [
            {"detector": "iforest", "precision": 0.8, "recall": 0.7, "f1": 0.75, "auc": 0.95, "n_flagged": 20},
            {"detector": "z-score", "precision": 0.5, "recall": 0.5, "f1": 0.5, "auc": 0.8, "n_flagged": 20},
        ]
        text = rep.comparison_table(rows)
        assert "detector" in text and "auc" in text
        assert "iforest" in text and "z-score" in text
        assert "0.9500" in text


class TestCaveats:
    def test_nonempty_and_headed(self):
        text = rep.caveats()
        assert text.startswith("## Caveats")
        assert len(text) > 200


class TestWriteReport:
    def test_writes_file(self, tmp_path):
        out = tmp_path / "sub" / "report.md"
        path = rep.write_report(str(out), "Anomaly Report", ["one", "two"])
        assert path == str(out)
        text = out.read_text(encoding="utf-8")
        assert text.startswith("# Anomaly Report")
        assert "one" in text and "two" in text

    def test_returns_path_string(self, tmp_path):
        path = rep.write_report(str(tmp_path / "r.md"), "T", ["a"])
        assert path.endswith("r.md")
