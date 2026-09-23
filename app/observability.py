"""
Observabilidad de la API de detección de defectos PCB.

La API escribe eventos JSON en stdout.

Cloud Run captura stdout automáticamente y los eventos
pueden consultarse posteriormente desde Cloud Logging.

No se guardan las imágenes ni su contenido.
"""

from __future__ import annotations

import json
import logging
import sys
import time

from contextlib import contextmanager
from typing import Any, Iterator


logger = logging.getLogger(
    "pcb_defect_detection.api"
)

_configured = False


def configure_logging() -> None:
    """Configura logging JSON hacia stdout."""

    global _configured

    if _configured:
        return

    handler = logging.StreamHandler(
        sys.stdout
    )

    handler.setFormatter(
        logging.Formatter("%(message)s")
    )

    logger.addHandler(handler)

    logger.setLevel(logging.INFO)

    logger.propagate = False

    _configured = True


@contextmanager
def measure_latency() -> Iterator[
    dict[str, float]
]:

    holder = {
        "latency_ms": 0.0
    }

    start = time.perf_counter()

    try:

        yield holder

    finally:

        holder["latency_ms"] = round(
            (
                time.perf_counter()
                - start
            )
            * 1000,
            2,
        )


def log_event(
    event: str,
    **fields: Any,
) -> None:

    payload = {
        "event": event,
        **fields,
    }

    logger.info(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )