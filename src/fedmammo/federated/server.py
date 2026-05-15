"""Server entrypoint: configure FedAvg (or any registered strategy) and run
``flwr.simulation.start_simulation``.

Server-side **centralized evaluation** is optional and wired through the
``evaluate_fn`` parameter of the strategy: after each round, the aggregated
parameters are loaded into a held-out model and tested against the global
test set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from flwr.server import ServerConfig

from fedmammo.configs.schema import ExperimentConfig
from fedmammo.datasets import MammographyDataset, build_dataloader, build_dataset
from fedmammo.evaluation import Evaluator
from fedmammo.federated.client import client_fn_factory
from fedmammo.federated.param_utils import (
    load_ndarrays_to_state_dict,
    state_dict_to_ndarrays,
)
from fedmammo.federated.strategies import build_strategy
from fedmammo.models import build_model
from fedmammo.training import build_loss
from fedmammo.utils.csv_logger import CSVLogger
from fedmammo.utils.device import resolve_device
from fedmammo.utils.logging_utils import get_logger
from fedmammo.utils.tensorboard_utils import TensorBoardWriter

_logger = get_logger(__name__)


def _build_evaluate_fn(
    cfg: ExperimentConfig,
    test_dataset: MammographyDataset | None,
    train_labels,  # noqa: ANN001 - numpy.ndarray
    tb_writer: TensorBoardWriter | None,
    csv_logger: CSVLogger | None,
) -> Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None] | None:
    """Build a centralized evaluate_fn that runs after each aggregation."""
    if test_dataset is None or len(test_dataset) == 0:
        return None

    device = resolve_device(cfg.device)
    template_model = build_model(cfg.model).to(device)
    criterion = build_loss(
        cfg.training.loss, train_labels=train_labels, num_classes=cfg.model.num_classes
    ).to(device)
    loader = build_dataloader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        balance_classes=False,
    )
    evaluator = Evaluator(template_model, device=device, threshold=cfg.evaluation.threshold)

    def evaluate_fn(
        server_round: int,
        parameters: NDArrays,
        config: dict[str, Scalar],  # noqa: ARG001
    ) -> tuple[float, dict[str, Scalar]]:
        load_ndarrays_to_state_dict(template_model, parameters, strict=True)
        result = evaluator.evaluate(loader, criterion=criterion)
        loss = float(result.get("loss", 0.0))
        numeric = {
            k: float(v)
            for k, v in result.items()
            if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v))
        }
        _logger.info(
            "[server] round %d centralized: loss=%.4f auc=%.4f f1=%.4f sens=%.4f spec=%.4f",
            server_round,
            loss,
            numeric.get("roc_auc", float("nan")),
            numeric.get("f1", float("nan")),
            numeric.get("sensitivity", float("nan")),
            numeric.get("specificity", float("nan")),
        )
        if tb_writer is not None:
            tb_writer.log_scalars("server/centralized", numeric, server_round)
        if csv_logger is not None:
            csv_logger.append({"round": server_round, "phase": "centralized", **numeric})
        # Flower expects (loss, metrics_dict). Drop non-numeric keys.
        return loss, {k: v for k, v in numeric.items() if k != "loss"}

    return evaluate_fn


def _initial_parameters(cfg: ExperimentConfig):  # noqa: ANN201
    """Build an untrained model and serialize it as Flower initial parameters."""
    device = resolve_device(cfg.device)
    model = build_model(cfg.model).to(device)
    ndarrays = state_dict_to_ndarrays(model)
    return fl.common.ndarrays_to_parameters(ndarrays)


def run_simulation(
    cfg: ExperimentConfig,
    *,
    output_dir: str | Path | None = None,
) -> Any:
    """Run a federated simulation according to ``cfg``.

    Args:
        cfg: Fully populated :class:`ExperimentConfig`.
        output_dir: Directory for TB / CSV artifacts. Defaults to
            ``<cfg.output_dir>/<cfg.name>``.

    Returns:
        The :class:`flwr.server.History` produced by ``start_simulation``.
    """
    out_root = Path(output_dir or Path(cfg.output_dir) / cfg.name).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    tb_writer = TensorBoardWriter(out_root / "tb")
    csv_logger = CSVLogger(out_root / "server_metrics.csv")

    _logger.info("Building datasets for federated simulation (%s)", cfg.data.name)
    datasets = build_dataset(cfg)

    # Strategy ----------------------------------------------------------
    strategy_kwargs: dict[str, Any] = {
        "fraction_fit": cfg.federated.fraction_fit,
        "fraction_evaluate": cfg.federated.fraction_evaluate,
        "min_fit_clients": cfg.federated.min_fit_clients,
        "min_evaluate_clients": cfg.federated.min_evaluate_clients,
        "min_available_clients": cfg.federated.min_available_clients,
        "accept_failures": cfg.federated.accept_failures,
        "initial_parameters": _initial_parameters(cfg),
        "evaluate_fn": _build_evaluate_fn(
            cfg,
            datasets.get("test"),
            datasets["train"].labels,
            tb_writer,
            csv_logger,
        ),
        "on_fit_config_fn": _make_on_fit_config_fn(cfg),
    }
    strategy_kwargs.update(cfg.federated.strategy.params)
    strategy = build_strategy(cfg.federated.strategy.name, **strategy_kwargs)

    client_fn = client_fn_factory(cfg, datasets)

    server_config = ServerConfig(num_rounds=cfg.federated.rounds)

    _logger.info(
        "Starting simulation: %d clients, %d rounds, strategy=%s",
        cfg.federated.num_clients,
        cfg.federated.rounds,
        cfg.federated.strategy.name,
    )
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.federated.num_clients,
        config=server_config,
        strategy=strategy,
        client_resources=cfg.federated.client_resources,
        ray_init_args=cfg.federated.ray_init_args or None,
    )
    tb_writer.close()
    _logger.info("Simulation complete. Artifacts in %s", out_root)
    return history


def _make_on_fit_config_fn(cfg: ExperimentConfig):
    """Return a per-round config function sent to clients in ``fit``."""

    def fn(server_round: int) -> dict[str, Scalar]:
        # Hook for per-round LR or local-epoch schedules; minimal default.
        return {
            "server_round": server_round,
            "local_epochs": cfg.training.local_epochs,
        }

    return fn


# ----------------------------------------------------------------------
# Real gRPC server (multi-device deployment)
# ----------------------------------------------------------------------

def run_grpc_server(
    cfg: ExperimentConfig,
    *,
    output_dir: str | Path | None = None,
) -> None:
    """Run the Flower gRPC server for a real multi-device deployment.

    Unlike :func:`run_simulation`, this function does not spawn clients and
    does not initialize Ray. It opens a gRPC port (``cfg.federated.server_address``)
    and waits for physical clients to connect. The clients must run
    ``scripts/run_client.py`` against this server's address.

    The server's local data (built from ``cfg``) is used exclusively for
    *centralized* evaluation via ``evaluate_fn``. In the canonical Mammo-Bench
    setup, the server YAML sets ``data.test_fraction=1.0`` and
    ``data.val_fraction=0.0`` so the entire local manifest becomes the held-out
    test set.

    Args:
        cfg: Fully populated :class:`ExperimentConfig`.
        output_dir: Directory for TB / CSV artifacts. Defaults to
            ``<cfg.output_dir>/<cfg.name>``.

    Returns:
        None. Blocks until ``cfg.federated.rounds`` rounds complete or the
        server is interrupted.
    """
    out_root = Path(output_dir or Path(cfg.output_dir) / cfg.name).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    tb_writer = TensorBoardWriter(out_root / "tb")
    csv_logger = CSVLogger(out_root / "server_metrics.csv")

    _logger.info(
        "Building server-side test dataset for centralized evaluation (%s)",
        cfg.data.name,
    )
    datasets = build_dataset(cfg)
    test_dataset = datasets.get("test")
    if test_dataset is None or len(test_dataset) == 0:
        # In gRPC mode the server has no train labels, but the loss factory
        # needs a fallback. Use test labels if available, otherwise None.
        _logger.warning(
            "Server build_dataset(cfg) produced no test split. Centralized "
            "evaluation will be disabled."
        )
        train_labels_for_loss = None
    else:
        train_labels_for_loss = test_dataset.labels

    strategy_kwargs: dict[str, Any] = {
        "fraction_fit": cfg.federated.fraction_fit,
        "fraction_evaluate": cfg.federated.fraction_evaluate,
        "min_fit_clients": cfg.federated.min_fit_clients,
        "min_evaluate_clients": cfg.federated.min_evaluate_clients,
        "min_available_clients": cfg.federated.min_available_clients,
        "accept_failures": cfg.federated.accept_failures,
        "initial_parameters": _initial_parameters(cfg),
        "evaluate_fn": _build_evaluate_fn(
            cfg,
            test_dataset,
            train_labels_for_loss,
            tb_writer,
            csv_logger,
        ),
        "on_fit_config_fn": _make_on_fit_config_fn(cfg),
    }
    strategy_kwargs.update(cfg.federated.strategy.params)
    strategy = build_strategy(cfg.federated.strategy.name, **strategy_kwargs)

    server_config = ServerConfig(num_rounds=cfg.federated.rounds)

    _logger.info(
        "Starting gRPC server on %s for %d rounds (strategy=%s, "
        "min_available_clients=%d). Waiting for clients to connect ...",
        cfg.federated.server_address,
        cfg.federated.rounds,
        cfg.federated.strategy.name,
        cfg.federated.min_available_clients,
    )
    try:
        fl.server.start_server(
            server_address=cfg.federated.server_address,
            config=server_config,
            strategy=strategy,
        )
    finally:
        tb_writer.close()
        _logger.info("gRPC server stopped. Artifacts in %s", out_root)


__all__ = ["run_simulation", "run_grpc_server"]
