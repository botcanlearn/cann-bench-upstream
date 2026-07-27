"""PyPTO-Pro artifact to Stanford/KernelBench submission converter."""

from __future__ import annotations

from pathlib import Path

from auto_pipeline.generator.pypto_pro.converter.base import PyptoProToBenchmarkConverter

_TEMPLATE_DIR = Path(__file__).with_name("templates")


class PyptoProToStanfordConverter(PyptoProToBenchmarkConverter):
    """Converts PyPTO-Pro artifacts into Stanford/KernelBench submissions."""

    name = "pypto-pro-to-stanford"
    target_benchmark = "stanford"
    conversion_template = _TEMPLATE_DIR / "to_stanford.j2"
