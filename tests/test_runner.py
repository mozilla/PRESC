import os
from subprocess import CompletedProcess
from pathlib import Path
import shutil

from pandas.testing import assert_frame_equal
import pytest
from sklearn.pipeline import Pipeline
import yaml

from presc.report.runner import Context, ReportRunner, load_config
from presc.utils import PrescError


@pytest.fixture
def webbrowser_patched(monkeypatch):
    import webbrowser

    def _print_url(url, *args, **kwargs):
        print(url)

    monkeypatch.setattr(webbrowser, "open", _print_url)
    monkeypatch.setattr(webbrowser, "open_new", _print_url)
    monkeypatch.setattr(webbrowser, "open_new_tab", _print_url)


def test_context(tmp_path, test_dataset, classification_model):
    ctx = Context(store_dir=tmp_path)
    ctx.store_inputs(test_dataset=test_dataset)

    # Data store was created.
    # The number of files and extensions are platform-dependent.
    ctx_store_paths = [p for p in tmp_path.iterdir() if p.stem == "_context_store"]
    assert len(ctx_store_paths) > 0

    # A new context instance should load from the same data store.
    ctx2 = Context(store_dir=tmp_path)
    td2 = ctx2.test_dataset
    assert_frame_equal(td2.df, test_dataset.df)
    with pytest.raises(PrescError):
        ctx2.model

    # Adding additional data to the store.
    config = {"bins": 10}
    ctx2.store_inputs(config=config, model=classification_model)

    assert ctx2.config == config
    assert isinstance(ctx2.model.classifier, Pipeline)
    assert_frame_equal(ctx.test_dataset.df, test_dataset.df)

    assert ctx.config == config
    assert isinstance(ctx.model.classifier, Pipeline)
    assert_frame_equal(ctx.test_dataset.df, test_dataset.df)


def test_config(tmp_path):
    dummy_config = {"a": 1, "b": {"c": ["x", "y", "z"], "d": True}}
    yaml_file = tmp_path / "config.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(dummy_config, f)
    not_yaml_file = tmp_path / "config.txt"
    with open(not_yaml_file, "w") as f:
        f.write("\t" + str(dummy_config))

    loaded_config = load_config(yaml_file)
    assert loaded_config == dummy_config
    with pytest.raises(PrescError):
        load_config(not_yaml_file)
    with pytest.raises(PrescError):
        load_config(tmp_path / "xyz.yaml")

    default_config = load_config()
    assert isinstance(default_config, dict)


def test_report_runner(tmp_path):
    # Check paths are initialized correctly by the runner
    rr = ReportRunner()
    assert str(rr.output_path.parent.resolve()) == os.getcwd()
    assert rr.output_path.exists()
    assert rr.execution_path is None
    assert isinstance(rr.config, dict)

    out_path = tmp_path / "abc" / "out"
    exec_path = tmp_path / "exec"
    conf_path = tmp_path / "conf.yaml"
    dummy_conf = {"a": 1}
    with open(conf_path, "w") as f:
        yaml.dump(dummy_conf, f)

    rr = ReportRunner(
        output_path=out_path, execution_path=exec_path, config_filepath=conf_path
    )
    assert rr.output_path.parent == out_path
    assert rr.output_path.exists()
    assert rr.execution_path.parent == exec_path
    assert rr.config == dummy_conf


def test_run_report(
    tmp_path, classification_model, test_dataset, webbrowser_patched, capsys
):
    out_path_run = tmp_path / "test_run"
    exec_path_run = tmp_path / "test_exec"
    rr = ReportRunner(output_path=out_path_run, execution_path=exec_path_run)
    # Run a report on the test data. This will take ~10 seconds
    rr.run(model=classification_model, test_dataset=test_dataset)

    # Check top-level output files exist and paths resolve
    assert isinstance(rr._jb_build_result, CompletedProcess)
    assert rr._jb_build_result.returncode == 0
    with open(rr.jb_build_log) as f:
        build_log = f.read()
    assert build_log.startswith("Running Jupyter-Book")
    assert "Finished generating HTML" in build_log
    assert "error" not in build_log.lower()
    assert "failed" not in build_log.lower()

    assert isinstance(rr._jb_clean_result, CompletedProcess)
    assert rr._jb_clean_result.returncode == 0
    # File should be empty as the output folder did not contain
    # a previous `_build` dir.
    assert rr.jb_clean_log.exists()

    assert rr.report_main_page.exists()
    # Link may not exist as it is platform-dependent.
    if rr._linked_main_page.exists():
        assert rr._linked_main_page.resolve() == rr.report_main_page.resolve()
        report_path = Path(rr.report_html)
        assert rr._linked_main_page.parent.resolve() == report_path.parent
        assert rr._linked_main_page.name == report_path.name
    else:
        assert rr.report_html == str(rr.report_main_page.resolve())

    # Opening in the browser: check the URL that is passed.
    capsys.readouterr()
    rr.open()
    url = capsys.readouterr().out.strip()
    assert url.startswith("file://")
    assert url.endswith(rr.report_html)

    # Check execution dir and output dir contains relevant files.
    assert len(list(rr.execution_path.glob("_context_store*"))) > 0
    assert (rr.execution_path / "_config.yml").exists()
    with open(rr.execution_path / "_toc.yml") as f:
        toc = [x.strip() for x in f.readlines()]
        notebooks = [x[8:] for x in toc if x.startswith("- file: ")]
    for nb in notebooks:
        assert (rr.execution_path / f"{nb}.ipynb").exists()
        assert (rr.report_main_page.parent / f"{nb}.html").exists()

    # Test cleaning on existing report output.
    rr.clean()
    assert not rr.jb_build_log.exists()
    assert not rr._linked_main_page.exists()
    assert not rr.report_main_page.parent.exists()
    assert isinstance(rr._jb_clean_result, CompletedProcess)
    assert rr._jb_clean_result.returncode == 0
    with open(rr.jb_clean_log) as f:
        clean_log = f.read()
    assert "Your _build dir" in clean_log
    assert "error" not in clean_log.lower()
    assert "failed" not in clean_log.lower()

    # Test rerunning report with the same runner.
    # Since the execution dir already exists, this tests that is get cleaned
    # successfully prior to running.
    rr.run(model=classification_model, test_dataset=test_dataset, clean=False)
    assert rr.report_main_page.parent.exists()
    assert rr._jb_build_result.returncode == 0


def test_run_report_tmp_exec_dir(tmp_path, classification_model, test_dataset):
    out_path_run = tmp_path / "test_run"
    rr = ReportRunner(output_path=out_path_run)
    # Run a report using the using default temp execution dir.
    # This will take ~10 seconds
    rr.run(model=classification_model, test_dataset=test_dataset)

    # Just check that it worked.
    assert rr.report_main_page.exists()
    assert rr._jb_build_result.returncode == 0


def test_run_report_error_notebook(tmp_path, pipeline_classifier, test_dataset):
    # Error encountered while running the notebooks
    out_path_run = tmp_path / "test_run"
    exec_path_run = tmp_path / "test_exec"
    rr = ReportRunner(output_path=out_path_run, execution_path=exec_path_run)
    # pipeline_classifier is not a valid ClassificationModel instance
    rr.run(model=pipeline_classifier, test_dataset=test_dataset)

    # jupyter-book build job succeeded even though notebooks didn't
    assert rr._jb_build_result.returncode == 0
    with open(rr.jb_build_log) as f:
        build_log = f.read()
    assert build_log.startswith("Running Jupyter-Book")
    assert "Finished generating HTML" in build_log
    # Log file mentions failure
    assert "failed" in build_log.lower()

    # Report was created
    assert rr.report_main_page.exists()
    # Execution dir and output dir contains relevant files.
    assert len(list(rr.execution_path.glob("_context_store*"))) > 0
    with open(rr.execution_path / "_toc.yml") as f:
        toc = [x.strip() for x in f.readlines()]
        notebooks = [x[8:] for x in toc if x.startswith("- file: ")]
    for nb in notebooks:
        assert (rr.execution_path / f"{nb}.ipynb").exists()
        assert (rr.report_main_page.parent / f"{nb}.html").exists()
    # Error message shows in notebooks
    with open(rr.report_main_page.parent / "landing.html") as f:
        landing_html = f.read()
    assert "AttributeError" in landing_html


def test_run_report_error_build(
    tmp_path, classification_model, test_dataset, monkeypatch
):
    # Missing report page: jupyter-book build job fails
    from presc.report import runner

    mock_report_source = tmp_path / "mock_report_source"
    shutil.copytree(runner.REPORT_SOURCE_PATH, mock_report_source)
    with open(mock_report_source / "_toc.yml", "a") as f:
        f.write("- file: missing_notebook\n")
    monkeypatch.setattr(runner, "REPORT_SOURCE_PATH", mock_report_source)

    out_path_run = tmp_path / "test_run"
    rr = ReportRunner(output_path=out_path_run)
    # run() function generates warnings
    with pytest.warns(UserWarning) as warning_records:
        rr.run(model=classification_model, test_dataset=test_dataset)
    assert len(warning_records) == 2
    # Warning from build function
    first_warning = warning_records[0].message.args[0]
    assert "jupyter-book build" in first_warning
    assert "did not succeed" in first_warning
    assert str(rr.jb_build_log) in first_warning
    # Warning from run function
    second_warning = warning_records[1].message.args[0]
    assert "expected report main page" in second_warning
    assert "error generating" in second_warning
    assert str(rr.jb_build_log) in second_warning

    # Build artifacts mention error
    assert rr._jb_build_result.returncode > 0
    with open(rr.jb_build_log) as f:
        build_log = f.read()
    assert build_log.startswith("Running Jupyter-Book")
    assert "Finished generating HTML" not in build_log
    assert "error in building" in build_log.lower()
    assert "RuntimeError" in build_log

    # Report was not produced
    assert not rr.report_main_page.exists()
    assert not rr._linked_main_page.exists()
    with pytest.raises(AttributeError):
        rr.report_html
    with pytest.raises(AttributeError):
        rr.open()


def test_run_report_error_config(tmp_path, classification_model, test_dataset):
    # Passing in config file to run() overrides default config.
    # Fake config causes error.
    out_path_run = tmp_path / "test_run"
    exec_path_run = tmp_path / "test_exec"
    fake_config_file = tmp_path / "fake_config.yaml"
    with open(fake_config_file, "w") as f:
        yaml.dump({"fake_option": 123}, f)

    rr = ReportRunner(output_path=out_path_run, execution_path=exec_path_run)
    rr.run(
        model=classification_model,
        test_dataset=test_dataset,
        config_filepath=fake_config_file,
    )

    # jupyter-book build job succeeded even though notebooks didn't
    assert rr._jb_build_result.returncode == 0
    with open(rr.jb_build_log) as f:
        build_log = f.read()
    assert build_log.startswith("Running Jupyter-Book")
    assert "Finished generating HTML" in build_log
    # Log file mentions failure
    assert "failed" in build_log.lower()

    # Report was created
    assert rr.report_main_page.exists()
    # Execution dir and output dir contains relevant files.
    assert len(list(rr.execution_path.glob("_context_store*"))) > 0
    with open(rr.execution_path / "_toc.yml") as f:
        toc = [x.strip() for x in f.readlines()]
        notebooks = [x[8:] for x in toc if x.startswith("- file: ")]
    for nb in notebooks:
        assert (rr.execution_path / f"{nb}.ipynb").exists()
        assert (rr.report_main_page.parent / f"{nb}.html").exists()
    # Error message shows in notebooks
    with open(rr.report_main_page.parent / "conditional_metric.html") as f:
        eval_html = f.read()
    assert "Error" in eval_html
    assert "Traceback" in eval_html
