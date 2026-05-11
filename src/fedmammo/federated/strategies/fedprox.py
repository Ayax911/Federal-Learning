"""FedProx strategy — STUB.

FedProx (Li et al., 2020) adds a proximal term ``(mu / 2) * ||w - w_global||^2``
to the client's local objective, which discourages large drifts from the
global model between communication rounds. The server-side aggregation is
identical to FedAvg, so the strategy class is essentially a marker that
configures the client (via the ``config`` dict in ``fit``) to apply the
proximal term locally.

Implementation plan when filled in:

1. Subclass :class:`flwr.server.strategy.FedAvg` (or reuse it directly).
2. In :meth:`configure_fit`, inject ``mu`` into each client's ``FitIns.config``.
3. On the client side (see :class:`fedmammo.federated.client.FedMammoClient`),
   add a proximal-term loss component when ``config.get("proximal_mu", 0.0) > 0``.

This stub registers the name so configs can reference ``fedprox`` without
breaking, and raises a clear error if a round is actually launched.
"""

from __future__ import annotations

from typing import Any

from flwr.common import FitIns, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fedmammo.federated.strategies.fedavg import _weighted_average
from fedmammo.federated.strategies.registry import register_strategy
from fedmammo.utils.logging_utils import get_logger

_logger = get_logger(__name__)


class FedProx(FedAvg):
    """FedProx server strategy (scaffold).

    Injects ``proximal_mu`` into the per-round client config so the client
    training loop can add the regularizer. Aggregation logic is identical to
    FedAvg.

    Args:
        proximal_mu: Strength of the proximal regularizer (Li et al., 2020).
    """

    def __init__(self, *, proximal_mu: float = 0.01, **kwargs: Any) -> None:
        kwargs.setdefault("fit_metrics_aggregation_fn", _weighted_average)
        kwargs.setdefault("evaluate_metrics_aggregation_fn", _weighted_average)
        super().__init__(**kwargs)
        self.proximal_mu = float(proximal_mu)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        # TODO(fedprox): once the client-side proximal term lands, this
        # implementation is sufficient. Until then we keep it functional but
        # warn so misuse is loud.
        instructions = super().configure_fit(server_round, parameters, client_manager)
        out: list[tuple[ClientProxy, FitIns]] = []
        for client, fit_ins in instructions:
            cfg: dict[str, Scalar] = dict(fit_ins.config)
            cfg["proximal_mu"] = self.proximal_mu
            out.append((client, FitIns(fit_ins.parameters, cfg)))
        if not getattr(self, "_warned_stub", False):
            _logger.warning(
                "FedProx is a scaffold: the proximal-term contribution to the "
                "client loss is not yet implemented. Behaves as FedAvg in this build."
            )
            self._warned_stub = True
        return out


@register_strategy("fedprox")
def build_fedprox(**kwargs: Any) -> FedAvg:
    return FedProx(**kwargs)


__all__ = ["FedProx", "build_fedprox"]
