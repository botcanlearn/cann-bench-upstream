"""PyPTO-Pro artifact to CANN benchmark submission converter."""

from __future__ import annotations

from pathlib import Path

from auto_pipeline.generator.pypto_pro.converter.base import PyptoProToBenchmarkConverter

_TEMPLATE_DIR = Path(__file__).with_name("templates")


class PyptoProToCannConverter(PyptoProToBenchmarkConverter):
    """Converts PyPTO-Pro artifacts into CANN benchmark submissions."""

    name = "pypto-pro-to-cann"
    target_benchmark = "cann"
    conversion_template = _TEMPLATE_DIR / "to_cann.j2"
