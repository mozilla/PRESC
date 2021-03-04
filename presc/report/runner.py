from pathlib import Path
import shelve
from tempfile import TemporaryDirectory
import shutil
import subprocess
import warnings
import webbrowser

import yaml

from presc.utils import PrescError

# Path to the report source dir
REPORT_SOURCE_PATH = Path(__file__).parent / "resources"
REPORT_OUTPUT_DIR = "presc_report"
REPORT_EXECUTION_DIR = "_exec"
REPORT_MAIN_PAGE = "report.html"
SPHINX_INDEX_PAGE = Path("_build") / "html" / "index.html"
JB_CLEAN_LOG = "jupyterbook_clean.log"
JB_BUILD_LOG = "jupyterbook_build.log"
# Path to the store for the inputs to the report, relative to the execution dir
# for the report.
CONTEXT_STORE_BASENAME = "_context_store"
# Path to the default config file
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config_default.yaml"


def load_config(config_filepath=None):
    """Loads the configuration from the given file.

    config_filepath: path to a YAML file listing PRESC config options

    Returns the config values as a dict.
    """
    if not config_filepath:
        config_filepath = DEFAULT_CONFIG_PATH
    try:
        with open(config_filepath) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
    except IOError as e:
        msg = f"Error opening config file {config_filepath}"
        raise PrescError(msg) from e
    except yaml.YAMLError as e:
        msg = f"Error parsing config file {config_filepath}"
        raise PrescError(msg) from e


class ReportRunner:
    """Main entrypoint to run the PRESC report for the given modeling inputs.

    The report is written to <output_path>/presc_report. If this dir already
    exists, it will be overwritten.

    To generate the report:
    ```
    pr = ReportRunner()
    pr.run(...)
    ```

    The path to the report is accessible as `pr.report_html` and will open in
    the default browser by calling `pr.open()`

    output_path: path to the dir where the report will be written. If not
        specified, defaults to the current working dir.
    execution_path: path from which the report is built. If not specified, a
        temporary dir is used.
    config_filepath: path to a YAML file listing config options to govern the
        report behaviour. If not specified, the default config is used.
    """

    def __init__(self, output_path=".", execution_path=None, config_filepath=None):
        # The main config options, as a dict.
        self.config = load_config(config_filepath)
        # Path where the report output is written.
        # Outputs are nested in a subdir.
        self.output_path = Path(output_path) / REPORT_OUTPUT_DIR
        self.output_path.mkdir(parents=True, exist_ok=True)
        # Path where the report is built from.
        # The report source files are copied here, and the model inputs are
        # written to a data store.
        # If missing, a temp dir will be used on execution.
        self.execution_path = None
        if execution_path is not None:
            self.execution_path = Path(execution_path) / REPORT_EXECUTION_DIR

        # Build artifacts:
        # The main entry page for the report.
        self.report_main_page = self.output_path / SPHINX_INDEX_PAGE
        # Log files for jupyter-book execution.
        self.jb_clean_log = self.output_path / JB_CLEAN_LOG
        self.jb_build_log = self.output_path / JB_BUILD_LOG
        # The main page will be linked to the top-level output dir, if possible.
        self._linked_main_page = self.output_path / REPORT_MAIN_PAGE
        # Cache the process results from running jupyter-book commands for
        # debugging.
        self._jb_clean_result = None
        self._jb_build_result = None

    def _presc_artifacts(self):
        """List of paths to remove from the top-level output dir on clean."""
        return [self._linked_main_page, self.jb_clean_log, self.jb_build_log]

    def run(self, model, test_dataset, config_filepath=None, clean=True):
        """Runs the PRESC report for the given modeling inputs.

        The report is written to <output_path>/presc_report. If this dir already
        exists, it will be overwritten.

        model: a pre-trained ClassificationModel instance to evaluate
        test_dataset: a test Dataset instance used to evaluate model performance
        config_filepath: path to a YAML file listing config options to govern
            the report behaviour. If specified, this overrides the current
            config.
        clean: should previous outputs be cleaned? Default: True
        """
        if config_filepath:
            config = load_config(config_filepath)
        else:
            config = self.config

        if clean:
            self.clean()

        tmpdir = None
        exec_path = None
        if self.execution_path:
            # If using a user-defined execution path, need to make sure
            # it doesn't exist for `shutil.copytree` to work.
            # Note that this will only remove the nested subdir, not the actual
            # user-specified dir.
            if self.execution_path.exists():
                shutil.rmtree(self.execution_path)
            exec_path = self.execution_path
        else:
            # Create a temp dir to run the build from.
            # We set up the temp dir here rather than using jupyter-book's
            # `run_in_temp` option so that we have access to the temp path.
            tmpdir = TemporaryDirectory()
            exec_path = Path(tmpdir.name) / REPORT_EXECUTION_DIR

        # Copy the report source files to the execution dir and
        # execute from there. The data store for the inputs is saved to
        # the same dir. That way, since the notebooks' working dir on execution
        # is set to where they are located by jupyter-book, they can find
        # the data store without needed to know the calling path.
        try:
            shutil.copytree(REPORT_SOURCE_PATH, exec_path)
        except shutil.Error as e:
            msg = f"Failed to copy report source to execution dir {exec_path}"
            raise PrescError(msg) from e
        # Write the inputs to the data store.
        ctx = Context(store_dir=exec_path)
        ctx.store_inputs(model=model, test_dataset=test_dataset, config=config)

        # Build the report.
        self._run_jb_build(exec_path)
        if tmpdir:
            tmpdir.cleanup()

        # The build should have created index.html at the `report_main_page`
        # path.
        if self.report_main_page.exists():
            # Symlink the main page to the top level for convenience.
            try:
                main_page_target = self.report_main_page.relative_to(
                    self._linked_main_page.parent
                )
                self._linked_main_page.symlink_to(main_page_target)
            except OSError:
                pass
        else:
            msg = f"The expected report main page {self.report_main_page} does not appear to exist."
            msg += " There may have been an error generating the report."
            msg += f" Output is written to {self.jb_build_log}"
            warnings.warn(msg)

    @property
    def report_html(self):
        """The main page of the HTML report."""
        # Return symlink, if available, for a more user-friendly experience.
        if self._linked_main_page.exists():
            # Resolve to an absolute path up to the symlink.
            report_path = (
                self._linked_main_page.parent.resolve() / self._linked_main_page.name
            )
        else:
            report_path = self.report_main_page.resolve()
        if not report_path.exists():
            msg = "Report file does not appear to exist."
            msg += " Make sure the report has already been built."
            raise AttributeError(msg)

        return str(report_path)

    def open(self):
        """Open the report in the default web browser."""
        webbrowser.open_new_tab(f"file://{self.report_html}")

    def clean(self):
        """Remove artifacts from a previous run, if any."""
        for p in self._presc_artifacts():
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        self._run_jb_clean()

    def _run_jb_clean(self):
        """Run `jupyter-book clean`."""
        with open(self.jb_clean_log, "w") as outfile:
            result = subprocess.run(
                ["jupyter-book", "clean", self.output_path],
                stdout=outfile,
                stderr=subprocess.STDOUT,
            )
        if result.returncode > 0:
            msg = f"`jupyter-book clean {self.output_path} did not succeed."
            msg += f" Output is written to {self.jb_clean_log}"
            warnings.warn(msg)

        self._jb_clean_result = result

    def _run_jb_build(self, input_path):
        """Run `jupyter-book build` on the given path."""
        with open(self.jb_build_log, "w") as outfile:
            result = subprocess.run(
                [
                    "jupyter-book",
                    "build",
                    "--path-output",
                    str(self.output_path.resolve()),
                    str(input_path.resolve()),
                ],
                stdout=outfile,
                stderr=subprocess.STDOUT,
            )
        if result.returncode > 0:
            msg = f"`jupyter-book build {input_path} did not succeed."
            msg += f" Output is written to {self.jb_build_log}"
            warnings.warn(msg)

        self._jb_build_result = result


class Context:
    """Persistent data store for sharing report inputs across notebooks.

    Note that the store implementation does not support concurrent access. It is
    up to the caller to ensure that multiple instances each have a unique store
    location.

    store_dir: the dir to contain the data store, implemented as one or more
        database files. If not specified, defaults to the current working dir.
    """

    def __init__(self, store_dir="."):
        store_path = Path(store_dir) / CONTEXT_STORE_BASENAME
        self._store_path = str(store_path.resolve())

    def store_inputs(self, model=None, test_dataset=None, config=None):
        """Write the report inputs to the data store.

        Any existing values will be overwritten.

        model: a ClassificationModel instance
        test_dataset: a Dataset instance
        config: a dict of config options
        """
        with shelve.open(self._store_path) as ctx:
            if model:
                ctx["model"] = model
            if test_dataset:
                ctx["test_dataset"] = test_dataset
            if config:
                ctx["config"] = config

    def _get(self, key):
        try:
            with shelve.open(self._store_path, flag="r") as ctx:
                val = ctx[key]
            return val
        except KeyError:
            raise PrescError(f"Could not find stored value for '{key}'")

    @property
    def model(self):
        return self._get("model")

    @property
    def test_dataset(self):
        return self._get("test_dataset")

    @property
    def config(self):
        return self._get("config")
