"""Synthetic mammography-like dataset for smoke tests and CI.

Generates deterministic uint8 noise arrays at run time. Labels are balanced
50/50 and the per-class signal is a faint mean shift so a model can
*nominally* learn — useful when checking that training loss decreases at all.
"""

from __future__ import annotations

import numpy as np

from fedmammo.datasets.base import MammographyDataset, Sample


class SyntheticMammographyDataset(MammographyDataset):
    """A deterministic synthetic dataset.

    Each sample has a unique ``(seed, index)`` pair so generated tensors are
    reproducible across runs but vary per index.

    Args:
        num_samples: Number of samples to generate.
        image_size: Square side length of the generated array.
        grayscale: If True, generate 1-channel images; else 3-channel.
        seed: Seed for the deterministic generator.
        positive_fraction: Fraction of samples assigned to the malignant class.
        signal_strength: How strongly the label biases pixel mean
            (set to 0.0 for pure noise; default gives a learnable signal).
        transform: Optional Albumentations transform.
    """

    def __init__(
        self,
        *,
        num_samples: int = 256,
        image_size: int = 224,
        grayscale: bool = True,
        seed: int = 0,
        positive_fraction: float = 0.5,
        signal_strength: float = 0.15,
        transform=None,  # noqa: ANN001
    ) -> None:
        rng = np.random.default_rng(seed)
        n_pos = int(round(num_samples * positive_fraction))
        labels = np.concatenate([np.zeros(num_samples - n_pos), np.ones(n_pos)]).astype(np.int64)
        rng.shuffle(labels)
        samples = [
            Sample(image_path=f"synthetic:{i}", label=int(labels[i]), patient_id=f"pid_{i}")
            for i in range(num_samples)
        ]
        super().__init__(
            samples=samples, transform=transform, image_format="png", grayscale=grayscale
        )
        self._image_size = int(image_size)
        self._seed = int(seed)
        self._signal_strength = float(signal_strength)
        self._grayscale = grayscale

    def load_image(self, path: str) -> np.ndarray:
        # path is "synthetic:<i>"; parse the index for a per-sample seed.
        try:
            idx = int(path.split(":", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Malformed synthetic path: {path!r}") from exc
        rng = np.random.default_rng(self._seed * 1_000_003 + idx)
        label = self.samples[idx].label
        h = w = self._image_size
        if self._grayscale:
            base = rng.normal(loc=128.0, scale=40.0, size=(h, w))
        else:
            base = rng.normal(loc=128.0, scale=40.0, size=(h, w, 3))
        # Add a faint, learnable label-dependent mean shift.
        base = base + (label * 2 - 1) * self._signal_strength * 40.0
        return np.clip(base, 0, 255).astype(np.uint8)


__all__ = ["SyntheticMammographyDataset"]
