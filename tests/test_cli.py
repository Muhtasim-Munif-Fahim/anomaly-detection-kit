from pathlib import Path

import numpy as np
import pytest

from anomaly_detection import cli


def _write_labelled_csv(path, X, y):
    header = ",".join([f"f{i}" for i in range(X.shape[1])] + ["y_true"])
    np.savetxt(path, np.column_stack([X, y]), delimiter=",", header=header, comments="", fmt="%.6f")


class TestSimulate:
    def test_writes_labelled_csv(self, tmp_path):
        out = tmp_path / "data.csv"
        rc = cli.main(
            ["simulate", "--n-samples", "200", "--n-features", "3", "--seed", "5", "--out", str(out)]
        )
        assert rc == 0
        text = out.read_text(encoding="utf-8")
        assert text.splitlines()[0] == "f0,f1,f2,y_true"
        data = np.loadtxt(out, delimiter=",", skiprows=1)
        assert data.shape == (200, 4)
        y = data[:, -1]
        assert int(y.sum()) == 10

    def test_outliers_flag(self, tmp_path):
        out = tmp_path / "d.csv"
        cli.main(["simulate", "--outliers", "scale", "--seed", "1", "--out", str(out)])
        data = np.loadtxt(out, delimiter=",", skiprows=1)
        assert int(data[:, -1].sum()) == 25


class TestDetect:
    def test_runs_all_detectors_on_labelled_data(self, tmp_path, capsys):
        csv_path = tmp_path / "data.csv"
        _write_labelled_csv(csv_path, np.random.default_rng(0).normal(size=(200, 4)), np.zeros(200))
        rc = cli.main(["detect", "--data", str(csv_path), "--contamination", "0.05", "--seed", "1"])
        assert rc == 0
        out = capsys.readouterr().out
        for name in ("isolation_forest", "lof", "copod", "hbos", "zscore", "mad", "iqr"):
            assert f"{name}: flagged" in out

    def test_prints_indices(self, tmp_path, capsys):
        csv_path = tmp_path / "data.csv"
        X = np.random.default_rng(1).normal(size=(50, 3))
        X[7] += 50.0
        _write_labelled_csv(csv_path, X, np.zeros(50))
        cli.main(["detect", "--data", str(csv_path), "--contamination", "0.05", "--indices", "3"])
        assert "flagged indices:" in capsys.readouterr().out

    def test_single_detector(self, tmp_path, capsys):
        csv_path = tmp_path / "data.csv"
        _write_labelled_csv(csv_path, np.zeros((50, 2)), np.zeros(50))
        cli.main(["detect", "--data", str(csv_path), "--detector", "mad", "--contamination", "0.1"])
        out = capsys.readouterr().out
        assert "mad: flagged" in out
        assert "lof: flagged" not in out

    def test_single_hbos_detector(self, tmp_path, capsys):
        csv_path = tmp_path / "data.csv"
        _write_labelled_csv(csv_path, np.random.default_rng(4).normal(size=(80, 3)), np.zeros(80))
        cli.main(["detect", "--data", str(csv_path), "--detector", "hbos", "--contamination", "0.1"])
        out = capsys.readouterr().out
        assert "hbos: flagged" in out
        assert "lof: flagged" not in out
        assert "copod: flagged" not in out


class TestEvaluate:
    def test_prints_comparison(self, tmp_path, capsys):
        csv_path = tmp_path / "data.csv"
        X = np.random.default_rng(2).normal(size=(200, 4))
        y = np.zeros(200)
        y[:10] = 1
        _write_labelled_csv(csv_path, X, y)
        rc = cli.main(["evaluate", "--data", str(csv_path), "--contamination", "0.05"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "detector" in out and "precision" in out
        assert "isolation_forest" in out
        assert "copod" in out
        assert "hbos" in out

    def test_writes_report_file(self, tmp_path, capsys):
        csv_path = tmp_path / "data.csv"
        _write_labelled_csv(csv_path, np.zeros((50, 2)), np.zeros(50))
        report_path = tmp_path / "cmp.md"
        cli.main(["evaluate", "--data", str(csv_path), "--contamination", "0.1", "--out", str(report_path)])
        assert report_path.exists()
        assert "# Detector comparison" in report_path.read_text(encoding="utf-8")

    def test_missing_y_true_fails(self, tmp_path):
        csv_path = tmp_path / "plain.csv"
        np.savetxt(csv_path, np.zeros((20, 3)), delimiter=",")
        with pytest.raises(SystemExit):
            cli.main(["evaluate", "--data", str(csv_path)])


class TestReport:
    def test_writes_full_report(self, tmp_path, capsys):
        out = tmp_path / "report.md"
        rc = cli.main(
            ["report", "--n-samples", "200", "--contamination", "0.05", "--seed", "3", "--out", str(out)]
        )
        assert rc == 0
        text = out.read_text(encoding="utf-8")
        assert text.startswith("# Anomaly detection report")
        for section in ("## Detector comparison", "## Score summaries", "## Threshold sweep", "## Caveats"):
            assert section in text
        assert "Best F1 threshold:" in text

    def test_prints_comparison_to_stdout(self, tmp_path, capsys):
        out = tmp_path / "r.md"
        cli.main(["report", "--n-samples", "100", "--seed", "1", "--out", str(out)])
        assert "detector" in capsys.readouterr().out


class TestParser:
    def test_unknown_detector_rejected(self):
        parser = cli.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["detect", "--data", "x.csv", "--detector", "bogus"])

    def test_unknown_command_rejected(self):
        parser = cli.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["frobnicate"])

    def test_detector_choices(self):
        assert cli.DETECTOR_NAMES == ["isolation_forest", "lof", "copod", "hbos", "zscore", "mad", "iqr"]
