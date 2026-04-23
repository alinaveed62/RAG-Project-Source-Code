"""Tests for run_all.py, the evaluation driver."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from evaluation.experiments.run_all import (
    EXPERIMENTS,
    ExperimentSpec,
    load_qa_pairs,
    main,
    parse_args,
    run_baseline,
    run_one,
    setup_logging,
)


@pytest.fixture
def qa_pairs_file(tmp_path):
    path = tmp_path / "qa.json"
    path.write_text(json.dumps([{"id": "q1", "question": "x?", "expected_answer": "y"}]))
    return path


@pytest.fixture
def results_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return d


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.results_dir == Path("evaluation/results")
        assert args.qa_pairs == Path("evaluation/test_set/qa_pairs.json")
        assert args.documents == Path("keats_scraper/data/processed/documents.jsonl")
        assert args.only is None
        assert args.skip is None
        assert args.force is False
        assert args.no_baseline is False
        assert args.no_figures is False

    def test_only_flag(self):
        args = parse_args(["--only", "llm_comparison", "topk_comparison"])
        assert args.only == ["llm_comparison", "topk_comparison"]

    def test_skip_flag(self):
        args = parse_args(["--skip", "llm_comparison"])
        assert args.skip == ["llm_comparison"]

    def test_force_flag(self):
        args = parse_args(["--force"])
        assert args.force is True

    def test_no_baseline_flag(self):
        args = parse_args(["--no-baseline"])
        assert args.no_baseline is True

    def test_no_figures_flag(self):
        args = parse_args(["--no-figures"])
        assert args.no_figures is True


class TestSetupLogging:
    def test_creates_log_file(self, tmp_path):
        root = logging.getLogger()
        prior_handlers = root.handlers[:]
        for h in prior_handlers:
            root.removeHandler(h)
        try:
            log_path = tmp_path / "dir" / "run.log"
            setup_logging(log_path)
            file_handlers = [
                h for h in root.handlers if isinstance(h, logging.FileHandler)
            ]
            assert any(Path(h.baseFilename) == log_path for h in file_handlers)
            assert log_path.parent.exists()
        finally:
            for h in root.handlers[:]:
                root.removeHandler(h)
                if isinstance(h, logging.FileHandler):
                    h.close()
            for h in prior_handlers:
                root.addHandler(h)


class TestLoadQaPairs:
    def test_reads_json(self, tmp_path):
        path = tmp_path / "qa.json"
        path.write_text(json.dumps([{"id": "a"}, {"id": "b"}]))
        assert load_qa_pairs(path) == [{"id": "a"}, {"id": "b"}]


def _fake_df(n: int = 1) -> pd.DataFrame:
    return pd.DataFrame({"x": list(range(n))})


def _spec(name: str, csv_name: str | None = None, needs_pipeline: bool = False, func=None):
    return ExperimentSpec(
        name=name,
        csv_name=csv_name or f"{name}.csv",
        needs_pipeline=needs_pipeline,
        func=func or MagicMock(return_value=_fake_df()),
    )


class TestRunOne:
    def test_skip_when_csv_exists_and_not_force(self, results_dir):
        spec = _spec("exp")
        csv_path = results_dir / "exp.csv"
        csv_path.write_text("a,b\n1,2\n")
        ok = run_one(
            spec,
            csv_path=csv_path,
            force=False,
            qa_pairs=[],
            config=MagicMock(),
            documents_path=Path("/dev/null"),
            pipeline=None,
        )
        assert ok is True
        spec.func.assert_not_called()

    def test_force_overrides_existing_csv(self, results_dir):
        spec = _spec("exp")
        csv_path = results_dir / "exp.csv"
        csv_path.write_text("old")
        ok = run_one(
            spec,
            csv_path=csv_path,
            force=True,
            qa_pairs=[],
            config=MagicMock(),
            documents_path=Path("/dev/null"),
            pipeline=None,
        )
        assert ok is True
        spec.func.assert_called_once()
        assert "x" in csv_path.read_text()

    def test_chunk_size_routed_with_documents_path(self, results_dir):
        func = MagicMock(return_value=_fake_df())
        spec = _spec("chunk_size_comparison", func=func, needs_pipeline=False)
        config = MagicMock()
        docs = Path("/tmp/docs.jsonl")
        run_one(
            spec,
            csv_path=results_dir / "chunk_size_comparison.csv",
            force=False,
            qa_pairs=[{"id": "q"}],
            config=config,
            documents_path=docs,
            pipeline=None,
        )
        func.assert_called_once_with(qa_pairs=[{"id": "q"}], documents_path=docs, config=config)

    def test_needs_pipeline_but_none_returns_false(self, results_dir, caplog):
        spec = _spec("topk_comparison", needs_pipeline=True)
        ok = run_one(
            spec,
            csv_path=results_dir / "topk_comparison.csv",
            force=False,
            qa_pairs=[],
            config=MagicMock(),
            documents_path=Path("/dev/null"),
            pipeline=None,
        )
        assert ok is False
        spec.func.assert_not_called()

    def test_needs_pipeline_passes_pipeline(self, results_dir):
        func = MagicMock(return_value=_fake_df())
        spec = _spec("topk_comparison", func=func, needs_pipeline=True)
        pipeline = MagicMock()
        run_one(
            spec,
            csv_path=results_dir / "topk_comparison.csv",
            force=False,
            qa_pairs=[{"id": "q"}],
            config=MagicMock(),
            documents_path=Path("/dev/null"),
            pipeline=pipeline,
        )
        func.assert_called_once_with(pipeline=pipeline, qa_pairs=[{"id": "q"}])

    def test_config_only_branch(self, results_dir):
        func = MagicMock(return_value=_fake_df())
        spec = _spec("embedding_comparison", func=func, needs_pipeline=False)
        config = MagicMock()
        run_one(
            spec,
            csv_path=results_dir / "embedding_comparison.csv",
            force=False,
            qa_pairs=[{"id": "q"}],
            config=config,
            documents_path=Path("/dev/null"),
            pipeline=None,
        )
        func.assert_called_once_with(qa_pairs=[{"id": "q"}], config=config)

    def test_exception_caught_returns_false(self, results_dir, caplog):
        spec = _spec("boom", func=MagicMock(side_effect=RuntimeError("no")))
        ok = run_one(
            spec,
            csv_path=results_dir / "boom.csv",
            force=False,
            qa_pairs=[],
            config=MagicMock(),
            documents_path=Path("/dev/null"),
            pipeline=None,
        )
        assert ok is False

    def test_non_dataframe_return_returns_false(self, results_dir):
        spec = _spec("notdf", func=MagicMock(return_value={"not": "a df"}))
        ok = run_one(
            spec,
            csv_path=results_dir / "notdf.csv",
            force=False,
            qa_pairs=[],
            config=MagicMock(),
            documents_path=Path("/dev/null"),
            pipeline=None,
        )
        assert ok is False

    def test_csv_written_on_success(self, results_dir):
        df = _fake_df(3)
        spec = _spec("good", func=MagicMock(return_value=df))
        csv_path = results_dir / "good.csv"
        ok = run_one(
            spec,
            csv_path=csv_path,
            force=False,
            qa_pairs=[],
            config=MagicMock(),
            documents_path=Path("/dev/null"),
            pipeline=None,
        )
        assert ok is True
        assert csv_path.exists()
        assert "0,1,2" in csv_path.read_text().replace("\n", ",")


class TestRunBaseline:
    def test_skip_when_exists_and_not_force(self, results_dir):
        out = results_dir / "baseline_results.json"
        out.write_text("{}")
        pipeline = MagicMock()
        ok = run_baseline(pipeline, qa_pairs=[], output_path=out, force=False)
        assert ok is True

    def test_evaluator_called_with_bert_and_sgf(self, results_dir):
        out = results_dir / "baseline_results.json"
        pipeline = MagicMock()
        with patch("evaluation.experiments.run_all.Evaluator") as MockEval:
            inst = MockEval.return_value
            inst.run.return_value = [{"id": "q"}]
            ok = run_baseline(pipeline, qa_pairs=[{"id": "q"}], output_path=out, force=False)
        assert ok is True
        inst.run.assert_called_once_with(compute_bert=True, with_sgf=True)

    def test_force_reruns_existing(self, results_dir):
        out = results_dir / "baseline_results.json"
        out.write_text("{}")
        pipeline = MagicMock()
        with patch("evaluation.experiments.run_all.Evaluator") as MockEval:
            MockEval.return_value.run.return_value = [{"id": "q"}]
            ok = run_baseline(pipeline, qa_pairs=[{"id": "q"}], output_path=out, force=True)
        assert ok is True
        assert MockEval.return_value.run.called

    def test_exception_returns_false(self, results_dir):
        out = results_dir / "baseline_results.json"
        pipeline = MagicMock()
        with patch("evaluation.experiments.run_all.Evaluator") as MockEval:
            MockEval.return_value.run.side_effect = RuntimeError("boom")
            ok = run_baseline(pipeline, qa_pairs=[], output_path=out, force=True)
        assert ok is False


def _fake_experiments(needs_pipeline_names: tuple[str, ...] = ()) -> list[ExperimentSpec]:
    """Build a list of fake ExperimentSpec objects with mocked func.

    Each fake spec matches a real one by name so the --only / --skip
    filtering logic exercises the same paths. The funcs are MagicMocks
    returning an empty DataFrame so the run writes a trivial CSV.
    """
    names = [
        ("embedding_comparison", False),
        ("chunk_size_comparison", False),
        ("retrieval_comparison", False),
        ("reranking_comparison", False),
        ("topk_comparison", True),
        ("llm_comparison", True),
        ("per_category_breakdown", True),
    ]
    out: list[ExperimentSpec] = []
    for name, default_needs in names:
        needs = name in needs_pipeline_names if needs_pipeline_names else default_needs
        out.append(
            ExperimentSpec(
                name=name,
                csv_name=f"{name}.csv",
                needs_pipeline=needs,
                func=MagicMock(return_value=_fake_df()),
            )
        )
    return out


class TestMain:
    def test_no_failures_no_pipeline_needed(self, tmp_path, qa_pairs_file):
        results_dir = tmp_path / "results"
        specs = _fake_experiments()
        with (
            patch("evaluation.experiments.run_all.EXPERIMENTS", specs),
            patch("evaluation.experiments.run_all.RAGPipeline") as MockPipeline,
            patch("evaluation.experiments.run_all.RAGConfig") as MockConfig,
            patch("evaluation.experiments.run_all.Evaluator"),
            patch("evaluation.experiments.run_all.generate_all_results") as mock_gen,
        ):
            MockConfig.return_value = MagicMock(
                embedding_model="m",
                ollama_model="g",
                top_k=3,
                enable_reranking=True,
                retrieval_mode="dense",
            )
            MockPipeline.return_value = MagicMock()
            rc = main(
                [
                    "--results-dir",
                    str(results_dir),
                    "--qa-pairs",
                    str(qa_pairs_file),
                    "--skip",
                    "topk_comparison",
                    "llm_comparison",
                    "per_category_breakdown",
                    "--no-baseline",
                ]
            )
        assert rc == 0
        # Pipeline was not constructed since nothing selected needed it.
        MockPipeline.assert_not_called()
        mock_gen.assert_called_once()

    def test_build_index_when_missing(self, tmp_path, qa_pairs_file):
        results_dir = tmp_path / "results"
        specs = _fake_experiments()
        with (
            patch("evaluation.experiments.run_all.EXPERIMENTS", specs),
            patch("evaluation.experiments.run_all.RAGPipeline") as MockPipeline,
            patch("evaluation.experiments.run_all.RAGConfig") as MockConfig,
            patch("evaluation.experiments.run_all.Evaluator") as MockEval,
            patch("evaluation.experiments.run_all.generate_all_results"),
        ):
            MockConfig.return_value = MagicMock(
                embedding_model="m",
                ollama_model="g",
                top_k=3,
                enable_reranking=True,
                retrieval_mode="dense",
                index_dir=tmp_path / "idx",
            )
            pipeline_instance = MagicMock()
            MockPipeline.return_value = pipeline_instance
            MockEval.return_value.run.return_value = [{"id": "q"}]
            rc = main(
                [
                    "--results-dir",
                    str(results_dir),
                    "--qa-pairs",
                    str(qa_pairs_file),
                    "--only",
                    "topk_comparison",
                ]
            )
        assert rc == 0
        pipeline_instance.build_index.assert_called_once()
        pipeline_instance.setup.assert_called_once()

    def test_existing_index_skips_build(self, tmp_path, qa_pairs_file):
        results_dir = tmp_path / "results"
        index_dir = tmp_path / "idx"
        index_dir.mkdir()
        (index_dir / "index.faiss").write_text("fake")
        specs = _fake_experiments()
        with (
            patch("evaluation.experiments.run_all.EXPERIMENTS", specs),
            patch("evaluation.experiments.run_all.RAGPipeline") as MockPipeline,
            patch("evaluation.experiments.run_all.RAGConfig") as MockConfig,
            patch("evaluation.experiments.run_all.Evaluator") as MockEval,
            patch("evaluation.experiments.run_all.generate_all_results"),
        ):
            MockConfig.return_value = MagicMock(
                embedding_model="m",
                ollama_model="g",
                top_k=3,
                enable_reranking=True,
                retrieval_mode="dense",
                index_dir=index_dir,
            )
            pipeline_instance = MagicMock()
            MockPipeline.return_value = pipeline_instance
            MockEval.return_value.run.return_value = []
            rc = main(
                [
                    "--results-dir",
                    str(results_dir),
                    "--qa-pairs",
                    str(qa_pairs_file),
                    "--only",
                    "topk_comparison",
                ]
            )
        assert rc == 0
        pipeline_instance.build_index.assert_not_called()

    def test_collects_failures_returns_1(self, tmp_path, qa_pairs_file):
        results_dir = tmp_path / "results"
        specs = _fake_experiments()
        specs[0].func.side_effect = RuntimeError("no")
        with (
            patch("evaluation.experiments.run_all.EXPERIMENTS", specs),
            patch("evaluation.experiments.run_all.RAGPipeline"),
            patch("evaluation.experiments.run_all.RAGConfig") as MockConfig,
            patch("evaluation.experiments.run_all.Evaluator"),
            patch("evaluation.experiments.run_all.generate_all_results"),
        ):
            MockConfig.return_value = MagicMock(
                embedding_model="m",
                ollama_model="g",
                top_k=3,
                enable_reranking=True,
                retrieval_mode="dense",
            )
            rc = main(
                [
                    "--results-dir",
                    str(results_dir),
                    "--qa-pairs",
                    str(qa_pairs_file),
                    "--only",
                    "embedding_comparison",
                    "--no-baseline",
                ]
            )
        assert rc == 1

    def test_generate_all_results_failure_caught(self, tmp_path, qa_pairs_file):
        results_dir = tmp_path / "results"
        specs = _fake_experiments()
        with (
            patch("evaluation.experiments.run_all.EXPERIMENTS", specs),
            patch("evaluation.experiments.run_all.RAGConfig") as MockConfig,
            patch(
                "evaluation.experiments.run_all.generate_all_results",
                side_effect=RuntimeError("boom"),
            ),
        ):
            MockConfig.return_value = MagicMock(
                embedding_model="m",
                ollama_model="g",
                top_k=3,
                enable_reranking=True,
                retrieval_mode="dense",
            )
            rc = main(
                [
                    "--results-dir",
                    str(results_dir),
                    "--qa-pairs",
                    str(qa_pairs_file),
                    "--only",
                    "nonexistent",
                    "--no-baseline",
                ]
            )
        assert rc == 1

    def test_no_figures_flag_skips_figure_generation(self, tmp_path, qa_pairs_file):
        results_dir = tmp_path / "results"
        specs = _fake_experiments()
        with (
            patch("evaluation.experiments.run_all.EXPERIMENTS", specs),
            patch("evaluation.experiments.run_all.RAGConfig") as MockConfig,
            patch(
                "evaluation.experiments.run_all.generate_all_results"
            ) as mock_gen,
        ):
            MockConfig.return_value = MagicMock(
                embedding_model="m",
                ollama_model="g",
                top_k=3,
                enable_reranking=True,
                retrieval_mode="dense",
            )
            main(
                [
                    "--results-dir",
                    str(results_dir),
                    "--qa-pairs",
                    str(qa_pairs_file),
                    "--only",
                    "xx",
                    "--no-baseline",
                    "--no-figures",
                ]
            )
        mock_gen.assert_not_called()

    def test_only_baseline_builds_pipeline_and_runs(self, tmp_path, qa_pairs_file):
        results_dir = tmp_path / "results"
        index_dir = tmp_path / "idx"
        index_dir.mkdir()
        (index_dir / "index.faiss").write_text("fake")
        specs = _fake_experiments()
        with (
            patch("evaluation.experiments.run_all.EXPERIMENTS", specs),
            patch("evaluation.experiments.run_all.RAGPipeline") as MockPipeline,
            patch("evaluation.experiments.run_all.RAGConfig") as MockConfig,
            patch("evaluation.experiments.run_all.Evaluator") as MockEval,
            patch("evaluation.experiments.run_all.generate_all_results"),
        ):
            MockConfig.return_value = MagicMock(
                embedding_model="m",
                ollama_model="g",
                top_k=3,
                enable_reranking=True,
                retrieval_mode="dense",
                index_dir=index_dir,
            )
            pipeline_instance = MagicMock()
            MockPipeline.return_value = pipeline_instance
            MockEval.return_value.run.return_value = []
            rc = main(
                [
                    "--results-dir",
                    str(results_dir),
                    "--qa-pairs",
                    str(qa_pairs_file),
                    "--only",
                    "baseline",
                ]
            )
        assert rc == 0
        pipeline_instance.setup.assert_called_once()

    def test_skip_flag_drops_experiments(self, tmp_path, qa_pairs_file):
        results_dir = tmp_path / "results"
        specs = _fake_experiments()
        with (
            patch("evaluation.experiments.run_all.EXPERIMENTS", specs),
            patch("evaluation.experiments.run_all.RAGPipeline"),
            patch("evaluation.experiments.run_all.RAGConfig") as MockConfig,
            patch("evaluation.experiments.run_all.Evaluator"),
            patch("evaluation.experiments.run_all.generate_all_results"),
        ):
            MockConfig.return_value = MagicMock(
                embedding_model="m",
                ollama_model="g",
                top_k=3,
                enable_reranking=True,
                retrieval_mode="dense",
            )
            main(
                [
                    "--results-dir",
                    str(results_dir),
                    "--qa-pairs",
                    str(qa_pairs_file),
                    "--skip",
                    "embedding_comparison",
                    "chunk_size_comparison",
                    "retrieval_comparison",
                    "reranking_comparison",
                    "topk_comparison",
                    "llm_comparison",
                    "per_category_breakdown",
                    "--no-baseline",
                ]
            )
        # None of the fake experiment funcs should have been invoked.
        for spec in specs:
            spec.func.assert_not_called()

    def test_baseline_failure_recorded(self, tmp_path, qa_pairs_file):
        """Cover the if not run_baseline(...) : failures.append('baseline') branch."""
        results_dir = tmp_path / "results"
        index_dir = tmp_path / "idx"
        index_dir.mkdir()
        (index_dir / "index.faiss").write_text("fake")
        specs = _fake_experiments()
        with (
            patch("evaluation.experiments.run_all.EXPERIMENTS", specs),
            patch("evaluation.experiments.run_all.RAGPipeline") as MockPipeline,
            patch("evaluation.experiments.run_all.RAGConfig") as MockConfig,
            patch("evaluation.experiments.run_all.Evaluator") as MockEval,
            patch("evaluation.experiments.run_all.generate_all_results"),
        ):
            MockConfig.return_value = MagicMock(
                embedding_model="m",
                ollama_model="g",
                top_k=3,
                enable_reranking=True,
                retrieval_mode="dense",
                index_dir=index_dir,
            )
            MockPipeline.return_value = MagicMock()
            MockEval.return_value.run.side_effect = RuntimeError("eval broke")
            rc = main(
                [
                    "--results-dir",
                    str(results_dir),
                    "--qa-pairs",
                    str(qa_pairs_file),
                    "--only",
                    "baseline",
                ]
            )
        assert rc == 1

    def test_pipeline_build_exception_propagates(self, tmp_path, qa_pairs_file):
        """If RAGPipeline construction raises, the exception is not swallowed by run_all."""
        results_dir = tmp_path / "results"
        specs = _fake_experiments()
        with (
            patch("evaluation.experiments.run_all.EXPERIMENTS", specs),
            patch(
                "evaluation.experiments.run_all.RAGPipeline",
                side_effect=RuntimeError("cant build"),
            ),
            patch("evaluation.experiments.run_all.RAGConfig") as MockConfig,
            patch("evaluation.experiments.run_all.generate_all_results"),
        ):
            MockConfig.return_value = MagicMock(
                embedding_model="m",
                ollama_model="g",
                top_k=3,
                enable_reranking=True,
                retrieval_mode="dense",
            )
            with pytest.raises(RuntimeError):
                main(
                    [
                        "--results-dir",
                        str(results_dir),
                        "--qa-pairs",
                        str(qa_pairs_file),
                        "--only",
                        "baseline",
                    ]
                )


class TestExperimentsRegistry:
    """Guard the 12-spec EXPERIMENTS registry so the driver stays in sync."""

    def test_all_eleven_names_present(self):
        expected = {
            "embedding_comparison",
            "chunk_size_comparison",
            "retrieval_comparison",
            "reranking_comparison",
            "topk_comparison",
            "llm_comparison",
            "per_category_breakdown",
            "chunking_strategy_comparison",
            "failure_modes",
            "latency_pareto",
            "significance_tests",
        }
        names = {s.name for s in EXPERIMENTS}
        assert names == expected

    def test_post_hoc_experiments_do_not_need_pipeline(self):
        post_hoc = {
            "chunking_strategy_comparison",
            "failure_modes",
            "latency_pareto",
            "significance_tests",
        }
        for spec in EXPERIMENTS:
            if spec.name in post_hoc:
                assert spec.needs_pipeline is False, spec.name


class TestRunOneNewDispatch:
    """Covers the four post-hoc dispatch branches added to run_one.

    Every branch exercises both the happy path (upstream inputs exist and
    the wrapped experiment is invoked with the correct kwargs) and the
    skip path (upstream missing -> warning logged, True returned so
    the driver does not flag the run as failed).
    """

    def _baseline_json(self, results_dir):
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "baseline_results.json").write_text(
            json.dumps([{"id": "q1", "is_refusal": False}]), encoding="utf-8"
        )
        return results_dir / "baseline_results.json"

    def test_failure_modes_dispatches_and_unpacks_tuple(self, results_dir):
        self._baseline_json(results_dir)
        per_query = pd.DataFrame({"id": ["q1"], "failure_mode": ["ok"]})
        summary = pd.DataFrame({"category": ["x"], "ok": [1]})
        func = MagicMock(return_value=(per_query, summary))
        spec = ExperimentSpec("failure_modes", "failure_modes.csv", False, func)
        csv_path = results_dir / "failure_modes.csv"
        ok = run_one(
            spec,
            csv_path=csv_path,
            force=True,
            qa_pairs=[],
            config=MagicMock(),
            documents_path=Path("/dev/null"),
            pipeline=None,
            results_dir=results_dir,
        )
        assert ok is True
        # The per-query DF is the one saved to csv_path; confirm the func
        # received the expected paths.
        call_kwargs = func.call_args.kwargs
        assert call_kwargs["baseline_json"] == results_dir / "baseline_results.json"
        assert call_kwargs["out_csv"] == csv_path
        assert call_kwargs["summary_csv"] == results_dir / "failure_mode_summary.csv"
        assert csv_path.exists()

    def test_failure_modes_skips_when_baseline_missing(self, results_dir, caplog):
        func = MagicMock()
        spec = ExperimentSpec("failure_modes", "failure_modes.csv", False, func)
        with caplog.at_level(logging.WARNING):
            ok = run_one(
                spec,
                csv_path=results_dir / "failure_modes.csv",
                force=True,
                qa_pairs=[],
                config=MagicMock(),
                documents_path=Path("/dev/null"),
                pipeline=None,
                results_dir=results_dir,
            )
        assert ok is True
        assert func.called is False
        assert "baseline_results.json" in caplog.text

    def test_latency_pareto_dispatches(self, results_dir):
        df = pd.DataFrame({"experiment": ["a"], "label": ["b"], "latency_ms": [1.0],
                           "quality": [0.5], "is_on_frontier": [True]})
        func = MagicMock(return_value=df)
        spec = ExperimentSpec("latency_pareto", "pareto_frontier.csv", False, func)
        csv_path = results_dir / "pareto_frontier.csv"
        ok = run_one(
            spec,
            csv_path=csv_path,
            force=True,
            qa_pairs=[],
            config=MagicMock(),
            documents_path=Path("/dev/null"),
            pipeline=None,
            results_dir=results_dir,
        )
        assert ok is True
        call_kwargs = func.call_args.kwargs
        assert call_kwargs["results_dir"] == results_dir
        assert call_kwargs["out_csv"] == csv_path
        assert csv_path.exists()

    def test_significance_tests_writes_via_write_outputs(self, results_dir):
        df = pd.DataFrame()
        func = MagicMock(return_value=df)
        spec = ExperimentSpec("significance_tests", "significance_tests.csv", False, func)
        csv_path = results_dir / "significance_tests.csv"
        with patch(
            "evaluation.experiments.run_all.write_significance_outputs"
        ) as mock_writer:
            ok = run_one(
                spec,
                csv_path=csv_path,
                force=True,
                qa_pairs=[],
                config=MagicMock(),
                documents_path=Path("/dev/null"),
                pipeline=None,
                results_dir=results_dir,
            )
        assert ok is True
        assert func.call_args.kwargs["per_query_dir"] == results_dir / "per_query"
        mock_writer.assert_called_once()
        assert mock_writer.call_args.kwargs["out_csv"] == csv_path

    def test_significance_tests_writer_exception_fails_run(self, results_dir, caplog):
        """write_significance_outputs errors must bubble up so the driver
        records a real failure rather than silently reporting success."""
        df = pd.DataFrame()
        func = MagicMock(return_value=df)
        spec = ExperimentSpec("significance_tests", "significance_tests.csv", False, func)
        csv_path = results_dir / "significance_tests.csv"
        with patch(
            "evaluation.experiments.run_all.write_significance_outputs",
            side_effect=RuntimeError("disk full"),
        ), caplog.at_level(logging.ERROR):
            ok = run_one(
                spec,
                csv_path=csv_path,
                force=True,
                qa_pairs=[],
                config=MagicMock(),
                documents_path=Path("/dev/null"),
                pipeline=None,
                results_dir=results_dir,
            )
        assert ok is False
        assert "disk full" in caplog.text

    def test_chunking_strategy_skips_when_semantic_missing(self, results_dir, caplog):
        func = MagicMock()
        spec = ExperimentSpec(
            "chunking_strategy_comparison",
            "chunking_strategy_comparison.csv",
            False,
            func,
        )
        config = MagicMock()
        config.chunks_path = Path("/dev/null-recursive")
        with caplog.at_level(logging.WARNING):
            ok = run_one(
                spec,
                csv_path=results_dir / "chunking_strategy_comparison.csv",
                force=True,
                qa_pairs=[],
                config=config,
                documents_path=Path("/dev/null"),
                pipeline=None,
                results_dir=results_dir,
            )
        assert ok is True
        assert func.called is False
        assert "semantic chunks missing" in caplog.text

    def test_chunking_strategy_dispatches_when_semantic_present(
        self, tmp_path, results_dir
    ):
        semantic_path = tmp_path / "keats_scraper" / "data" / "chunks" / \
            "chunks_for_embedding_semantic.jsonl"
        semantic_path.parent.mkdir(parents=True, exist_ok=True)
        semantic_path.write_text("{}\n")
        df = pd.DataFrame({"strategy": ["recursive"], "n_chunks": [379]})
        func = MagicMock(return_value=df)
        spec = ExperimentSpec(
            "chunking_strategy_comparison",
            "chunking_strategy_comparison.csv",
            False,
            func,
        )
        csv_path = results_dir / "chunking_strategy_comparison.csv"
        config = MagicMock()
        config.chunks_path = tmp_path / "keats_scraper" / "data" / "chunks" / \
            "chunks_for_embedding.jsonl"
        # Patch the hardcoded semantic path inside run_one to point at our fixture.
        with patch(
            "evaluation.experiments.run_all.Path",
            side_effect=lambda p: semantic_path if "semantic" in str(p) else Path(p),
        ):
            ok = run_one(
                spec,
                csv_path=csv_path,
                force=True,
                qa_pairs=[],
                config=config,
                documents_path=Path("/dev/null"),
                pipeline=None,
                results_dir=results_dir,
            )
        assert ok is True
        assert func.called is True
        assert csv_path.exists()
