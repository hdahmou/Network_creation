"""Exercise the diagnostic notebook's lightweight cells without loading an artifact."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import nbformat
import numpy as np
import pytest
import torch

from network_creation.model_inversion.posterior import PosteriorArtifact
from tests.model_inversion.support import artifact_metadata


@pytest.fixture(scope="module")
def notebook():
    path = Path(__file__).resolve().parents[2] / "notebooks/posterior_recovery_diagnostics.ipynb"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cells(notebook):
    return {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }


@pytest.fixture
def namespace(cells):
    # Load imports, constants and helpers, but skip styling and filesystem setup.
    definitions = [
        node for node in ast.parse(cells["8804fe84"]).body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        or (
            isinstance(node, ast.Assign)
            and all(isinstance(target, ast.Name) and target.id.isupper() for target in node.targets)
        )
    ]
    namespace = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "notebook setup", "exec"), namespace)
    exec(cells["posterior-map-truth-helpers"], namespace)
    return namespace


def test_notebook_is_valid_and_all_code_cells_compile(notebook, cells):
    nbformat.validate(nbformat.from_dict(notebook))
    for cell_id, source in cells.items():
        compile(source, cell_id, "exec")


@pytest.mark.parametrize("sample_size", [None, 1, 7, 100])
def test_case_selection_is_reproducible_stratified_and_without_replacement(
    cells, namespace, sample_size,
):
    groups = np.repeat([10, 20, 30], [1, 5, 14]).astype(np.float32)
    namespace.update(diagnostic_n_agents=groups, N_DIAGNOSTIC_CASES=sample_size, SEED=42)
    exec(cells["diagnostic-case-selection"], namespace)
    indices = namespace["diagnostic_indices"].copy()
    exec(cells["diagnostic-case-selection"], namespace)
    np.testing.assert_array_equal(indices, namespace["diagnostic_indices"])
    expected_size = len(groups) if sample_size is None else min(sample_size, len(groups))
    assert indices.dtype == np.int64
    assert len(indices) == len(np.unique(indices)) == expected_size
    assert np.all((0 <= indices) & (indices < len(groups)))
    counts = np.array([np.count_nonzero(groups[indices] == group) for group in [10, 20, 30]])
    quotas = np.array([1, 5, 14]) * expected_size / len(groups)
    assert np.all((counts == np.floor(quotas)) | (counts == np.ceil(quotas)))
    if expected_size == len(groups):
        np.testing.assert_array_equal(np.sort(indices), np.arange(len(groups)))


@pytest.mark.parametrize("groups, sample_size", [([], None), ([10], 0), ([10], -1)])
def test_case_selection_rejects_empty_or_nonpositive_requests(cells, namespace, groups, sample_size):
    namespace.update(
        diagnostic_n_agents=np.asarray(groups), N_DIAGNOSTIC_CASES=sample_size, SEED=42,
    )
    with pytest.raises(ValueError, match="must be positive"):
        exec(cells["diagnostic-case-selection"], namespace)


@pytest.mark.parametrize(
    "truth, prediction, categories, expected",
    [
        ([10, 10, 20, 20], [10, 20, 20, 20], [20, 10, 30],
         [[1, 0.5, 0], [0, 0.5, 0], [0, 0, 0]]),
        ([10.00001, 20, np.nan, np.inf, 10, 999], [20.00001, 20, 10, 10, 999, 10],
         [20, 10, 30], [[1, 1, 0], [0, 0, 0], [0, 0, 0]]),
        # Approximate matches select the first stored category, not the nearest.
        ([100000], [100000], [100001, 100000, 5], [[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ([], [], [10, 20], [[0, 0], [0, 0]]),
        ([999, np.nan], [10, 20], [10, 20], [[0, 0], [0, 0]]),
        ([], [], [], np.zeros((0, 0))),
    ],
)
def test_confusion_rates_preserve_orientation_matching_and_empty_categories(
    namespace, truth, prediction, categories, expected,
):
    rates = namespace["categorical_confusion_rates"](truth, prediction, categories)
    assert rates.dtype == np.float32
    np.testing.assert_allclose(rates, expected)


def test_confusion_rates_reject_misaligned_pairs(namespace):
    with pytest.raises(ValueError):
        namespace["categorical_confusion_rates"]([10, 20], [10], [10, 20])


def test_confusion_plot_preserves_labels_orientation_and_shared_colorbar(namespace):
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()
    image = namespace["plot_categorical_confusion"](
        ax, [10, 10, 20, 20], [10, 20, 20, 20], [20, 10, 30],
    )
    np.testing.assert_allclose(image.get_array(), [[1, 0.5, 0], [0, 0.5, 0], [0, 0, 0]])
    assert image.origin == "lower"
    assert image.get_clim() == (0, 1)
    assert not ax.yaxis_inverted()
    assert [label.get_text() for label in ax.get_xticklabels()] == ["20", "10", "30"]
    assert [label.get_text() for label in ax.get_yticklabels()] == ["20", "10", "30"]
    assert len(fig.axes) == 1
    colorbar = fig.colorbar(image, ax=ax)
    assert colorbar.mappable is image
    canvas.draw()


@pytest.mark.parametrize("num_samples, chunk_size", [(5, 2), (4, 2), (1, 4)])
def test_sampling_cell_matches_previous_per_case_seeds_and_samples(
    cells, namespace, num_samples, chunk_size,
):
    calls = []

    class RecordingPosterior:
        def sample(self, sample_shape, x, **kwargs):
            calls.append((sample_shape, x.tolist(), torch.initial_seed(), kwargs))
            return torch.column_stack((
                torch.rand(sample_shape[0]) + x[0, 0],
                torch.full(sample_shape, 10.0),
            ))

    item = PosteriorArtifact(
        Path("."), RecordingPosterior(), artifact_metadata(),
        np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int64),
    )
    x = np.array([[0.1, 1, 0], [0.2, 0, 1]], dtype=np.float32)
    theta = np.array([[0.1, 10], [0.2, 20]], dtype=np.float32)
    indices = np.array([1, 0])
    exec(cells["2a74fcf7"], namespace)
    namespace.update(
        artifact=item, diagnostic_data=SimpleNamespace(theta=theta, x=x),
        diagnostic_indices=indices, diagnostic_n_agents=np.array([10, 20]),
        N_POSTERIOR_SAMPLES_PER_CASE=num_samples, POSTERIOR_SAMPLE_CHUNK_SIZE=chunk_size,
        display=Mock(),
    )
    chunks_per_case = (num_samples + chunk_size - 1) // chunk_size
    expected = np.stack([
        item.sample_cases(
            observation,
            num_samples=num_samples,
            seed=namespace["SEED"] + case_index * chunks_per_case,
            device=namespace["DEVICE"],
            max_samples_per_call=chunk_size,
            max_sampling_batch_size=namespace["MAX_SAMPLING_BATCH_SIZE"],
            max_sampling_time=namespace["MAX_SAMPLING_TIME_SECONDS"],
            show_progress_bars=False,
        )[0]
        for case_index, observation in enumerate(x[indices])
    ])
    expected_calls = calls.copy()
    calls.clear()

    exec(cells["sample-diagnostic-posteriors"], namespace)

    np.testing.assert_array_equal(namespace["diagnostic_posterior_samples"], expected)
    assert namespace["diagnostic_posterior_samples"].dtype == np.float32
    np.testing.assert_array_equal(namespace["diagnostic_theta"], theta[indices])
    np.testing.assert_array_equal(namespace["selected_n_agents"], [20, 10])
    assert calls == expected_calls
    assert [call[2] for call in calls] == list(range(42, 42 + 2 * chunks_per_case))


@pytest.mark.parametrize("subplot_kw", [None, {"facecolor": "pink", "box_aspect": 2}])
def test_square_grid_preserves_geometry_and_subplot_options(namespace, subplot_kw):
    plt = namespace["plt"]
    fig, axes = namespace["square_panel_grid"](
        2, 3, subplot_kw=subplot_kw, sharex="col", sharey="col",
    )
    try:
        assert axes.shape == (2, 3)
        np.testing.assert_allclose(fig.get_size_inches(), [9.5, 6.75])
        for ax in axes.flat:
            assert ax.get_box_aspect() == 1
            position = ax.get_position()
            np.testing.assert_allclose(
                [position.width * 9.5, position.height * 6.75], [2.5, 2.5],
            )
        assert axes[0, 0].get_shared_x_axes().joined(axes[0, 0], axes[1, 0])
        if subplot_kw is not None:
            assert subplot_kw == {"facecolor": "pink", "box_aspect": 2}
            assert axes[0, 0].get_facecolor() != axes[0, 0].figure.get_facecolor()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("mixed", [False, True])
def test_recovery_matrix_preserves_annotations_layout_and_percent_colorbar(namespace, mixed):
    overrides = {} if mixed else {"discrete_theta_names": [], "discrete_theta_values": []}
    metadata = artifact_metadata(dataset_overrides=overrides)
    n_theta = len(metadata.dataset.theta_order)
    item = PosteriorArtifact(
        Path("."), object(), metadata,
        np.empty((0, n_theta), dtype=np.float32), np.empty(0, dtype=np.int64),
    )
    truth = np.array([[0.1, 10], [0.3, 20], [0.6, 10], [0.8, 20]], dtype=np.float32)
    estimates = np.array([[0.2, 10], [0.4, 10], [0.7, 20], [0.9, 10]], dtype=np.float32)
    plt = namespace["plt"]
    with plt.rc_context({"text.usetex": False, "axes.grid": False}):
        fig, axes = namespace["plot_recovery_matrix"](
            item, theta_names=item.theta_order, true_theta=truth[:, :n_theta],
            posterior_estimates=estimates[:, :n_theta], n_agents=np.array([10, 10, 20, 20]),
        )
        try:
            assert axes.shape == (2, n_theta)
            assert len(fig.axes) == 2 * n_theta + int(mixed)
            for ax in axes.flat:
                assert len(ax.texts) == 1
                assert ax.texts[0].get_position() == (0.04, 0.96)
                assert ax.texts[0].get_bbox_patch().get_alpha() == 0.75
                assert ax.get_box_aspect() == 1
            for ax in axes[:, 0]:
                assert ax.texts[0].get_text() == "r = 1.00"
                np.testing.assert_allclose(ax.get_xlim(), [-0.01, 1.01])
                np.testing.assert_allclose(ax.get_ylim(), [-0.01, 1.01])
                assert any(line.get_visible() for line in ax.get_xgridlines())
            if mixed:
                assert axes[0, 1].texts[0].get_text() == "Accuracy = 0.50"
                assert axes[1, 1].texts[0].get_text() == "Accuracy = 0.00"
                assert not any(line.get_visible() for line in axes[0, 1].get_xgridlines())
                formatter = fig.axes[-1].yaxis.get_major_formatter()
                assert isinstance(formatter, namespace["PercentFormatter"])
                assert [formatter(v) for v in np.linspace(0, 1, 5)] == [
                    "0%", "25%", "50%", "75%", "100%",
                ]
                with plt.rc_context({"text.usetex": True}):
                    assert formatter(0.25) == r"25\%"
            FigureCanvasAgg(fig).draw()
        finally:
            plt.close(fig)


@pytest.mark.parametrize("representation", ["final", "timeseries"])
@pytest.mark.parametrize("input_kind", ["discrete", "continuous", "missing"])
def test_loading_cell_decodes_inputs_and_configures_shared_exports(
    cells, namespace, tmp_path, representation, input_kind,
):
    overrides = {}
    x = np.array([[0.1, 1, 0], [0.2, 0, 1]], dtype=np.float32)
    if input_kind != "discrete":
        continuous = input_kind == "continuous"
        overrides.update(
            discrete_input_names=[], discrete_input_values=[],
            continuous_input_names=["n_agents"] if continuous else [],
            x_components=["density.mean", "n_agents"] if continuous else ["density.mean"],
        )
        x = np.array([[0.1, 10], [0.2, 20]], dtype=np.float32)
        if not continuous:
            x = x[:, :1]
    if representation == "timeseries":
        x = np.repeat(x[:, None, :], 2, axis=1)
        overrides.update(x_representation="timeseries", step_labels=[0.0, 0.1])
    overrides["x_event_shape"] = x.shape[1:]
    metadata = artifact_metadata(dataset_overrides=overrides)
    artifact_dir = tmp_path / "saved-posterior"
    artifact_dir.mkdir()
    item = PosteriorArtifact(
        artifact_dir, object(), metadata,
        np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int64),
    )
    np.savez_compressed(
        artifact_dir / metadata.sbi.diagnostic_data_file,
        theta=np.array([[0.1, 10], [0.2, 20]], dtype=np.float32), x=x,
        source_sample_ids=np.array([3, 5]),
        continuous_theta_names=np.array(item.continuous_theta_names),
        discrete_theta_names=np.array(item.discrete_theta_names),
        x_components=np.array(item.x_components),
    )
    namespace.update(
        project_root=tmp_path, ARTIFACT_DIR=artifact_dir.name,
        resolve_artifact_dir=Mock(return_value=artifact_dir),
        load_posterior_artifact=Mock(return_value=item), display=Mock(),
    )
    if input_kind == "missing":
        with pytest.raises(ValueError, match="require n_agents"):
            exec(cells["ab8d9659"], namespace)
        return

    exec(cells["ab8d9659"], namespace)

    values = namespace["diagnostic_n_agents"]
    assert values is namespace["diagnostic_input_values"]["n_agents"]
    assert values.dtype == np.float32
    assert values.shape == (2,)
    np.testing.assert_array_equal(values, [10, 20])
    output_dir = tmp_path / "artifacts" / "figures" / artifact_dir.name
    assert namespace["figure_output_dir"] == output_dir
    assert output_dir.is_dir()
    fig = Mock()
    namespace["fig"] = fig
    for cell_id, filename in [
        ("8a67a8aa", "sbc_curve.pdf"),
        ("export-sbc-by-agents-pdf", "sbc_curves_by_n_agents.pdf"),
        ("export-recovery-matrix-pdf", "posterior_parameter_recovery_matrix.pdf"),
    ]:
        exec(cells[cell_id], namespace)
        fig.savefig.assert_called_with(output_dir / filename, format="pdf", bbox_inches="tight")


@pytest.mark.parametrize("metrics", [
    {"train_loss": [3.456789, 1.25, 2.0], "validation_loss": [4.0, 2.0, 3.0]},
    {"train_loss": [-1.5, -2.5]},
    {"validation_loss": [0.125]},
    {},
    {"train_loss": [], "validation_loss": []},
])
def test_learning_curve_exports_its_summary_and_handles_missing_curves(
    cells, namespace, tmp_path, monkeypatch, metrics,
):
    metadata = artifact_metadata()
    (tmp_path / metadata.sbi.training_metrics_file).write_text(json.dumps(metrics), encoding="utf-8")
    previous_figure = Mock()
    namespace.update(
        artifact_dir=tmp_path, metadata=metadata, project_root=tmp_path,
        figure_output_dir=tmp_path, display=Mock(), fig=previous_figure,
    )
    plt = namespace["plt"]
    monkeypatch.setattr(plt, "show", Mock())
    with plt.style.context("science"), plt.rc_context({**namespace["FIGURE_STYLE"], "text.usetex": False}):
        exec(cells["learning-curve"], namespace)
        curves = {name: values for name, values in metrics.items() if values}
        pdf_path = tmp_path / "learning_curve.pdf"
        if not curves:
            assert not pdf_path.exists()
            previous_figure.savefig.assert_not_called()
            assert namespace["display"].call_args.args[0].empty
            plt.show.assert_not_called()
            return

        fig = namespace["fig"]
        try:
            assert pdf_path.read_bytes().startswith(b"%PDF-")
            assert len(fig.axes) == 2
            ax, summary_ax = fig.axes
            assert ax.get_ylabel() == "Loss"
            assert not summary_ax.axison
            assert len(ax.lines) == len(curves)
            table = namespace["summary_table"]
            assert [table[0, col].get_text().get_text() for col in range(5)] == [
                "Metric", "Epochs", "First", "Last", "Best (min)",
            ]
            for row_index, (name, values) in enumerate(curves.items(), start=1):
                line = ax.lines[row_index - 1]
                np.testing.assert_array_equal(line.get_ydata(), values)
                np.testing.assert_array_equal(line.get_xdata(), np.arange(1, len(values) + 1))
                assert [table[row_index, col].get_text().get_text() for col in range(5)] == [
                    namespace["curve_labels"][name], str(len(values)),
                    f"{values[0]:.6g}", f"{values[-1]:.6g}", f"{min(values):.6g}",
                ]
            displayed = namespace["display"].call_args.args[0]
            assert displayed.to_dict("records") == namespace["loss_summary"]
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            table_bounds = table.get_window_extent(renderer)
            assert table_bounds.y1 < ax.get_tightbbox(renderer).y0
            assert fig.bbox.contains(table_bounds.x0, table_bounds.y0)
            assert fig.bbox.contains(table_bounds.x1, table_bounds.y1)
            plt.show.assert_called_once()
        finally:
            plt.close(fig)


def test_ridgeline_cells_export_overall_and_each_group_before_show(
    cells, namespace, tmp_path, monkeypatch,
):
    item = PosteriorArtifact(
        tmp_path, object(), artifact_metadata(prior_sample_count=4),
        np.array([[0.1, 10], [0.3, 20], [0.6, 10], [0.8, 20]], dtype=np.float32),
        np.arange(4),
    )
    truth = item.prior_samples.copy()
    posterior = np.repeat(truth[:, None, :], 4, axis=1)
    plot = Mock(wraps=namespace["plot_posterior_ridgelines"])
    namespace.update(
        artifact=item, prior_samples=item.prior_samples, diagnostic_theta=truth,
        diagnostic_posterior_samples=posterior, RIDGELINE_CASES=2,
        selected_n_agents=np.array([10, 20, 10, 10]), n_agent_values=np.array([10, 20]),
        project_root=tmp_path, figure_output_dir=tmp_path, display=Mock(),
        plot_posterior_ridgelines=plot,
    )
    plt = namespace["plt"]
    saved = []

    def check_export_before_show():
        path = namespace["figure_path"]
        assert path.read_bytes().startswith(b"%PDF-")
        saved.append(path.name)
        plt.close(namespace["fig"])

    monkeypatch.setattr(plt, "show", check_export_before_show)
    existing_figures = set(plt.get_fignums())
    with plt.style.context("science"), plt.rc_context({"text.usetex": False}):
        try:
            exec(cells["ridgelines-overall"], namespace)
            exec(cells["ridgelines-by-agents"], namespace)
            assert saved == [
                "posterior_ridgelines.pdf",
                "posterior_ridgelines_n_agents_10.pdf",
                "posterior_ridgelines_n_agents_20.pdf",
            ]
            assert plot.call_count == 3
            for call, indices in zip(plot.call_args_list, [[0, 1], [0, 2], [1]], strict=True):
                np.testing.assert_array_equal(call.kwargs["true_theta"], truth[indices])
                np.testing.assert_array_equal(call.kwargs["posterior_samples"], posterior[indices])
        finally:
            for number in set(plt.get_fignums()) - existing_figures:
                plt.close(number)
