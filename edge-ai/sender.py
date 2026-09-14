#!/usr/bin/env python3

from azure.servicebus import ServiceBusClient, ServiceBusMessage
from io import BytesIO
from urllib.parse import urlparse
import config
import json
import logging
import os
import pandas as pd
import state_utils
import sys
import time
import uuid
import numpy as np


def _diag_log(enabled, message):
    if enabled:
        logging.info(f"[send-diagnostics] {message}")


def send(data, verbose_diagnostics=False, diagnostics_context=None):
    payload = json.dumps(data, indent=4)
    context = diagnostics_context or {}
    endpoint = urlparse(config.connstr).netloc or "unknown"

    _diag_log(
        verbose_diagnostics,
        "Preparing payload: "
        f"chars={len(payload)}, keys={sorted(list(data.keys()))[:8]}, "
        f"queue={config.queue_name}, endpoint={endpoint}, context={context}",
    )

    logging.info("Sending data to the dashboard ..")

    try:
        send_start = time.time()
        with ServiceBusClient.from_connection_string(config.connstr) as client:
            _diag_log(
                verbose_diagnostics,
                f"ServiceBusClient created for endpoint={endpoint}",
            )
            with client.get_queue_sender(config.queue_name) as sender:
                _diag_log(
                    verbose_diagnostics,
                    f"Queue sender selected: queue_name={config.queue_name}",
                )
                logging.info(payload)

                # Sending a single message
                single_message = ServiceBusMessage(payload)
                _diag_log(
                    verbose_diagnostics,
                    "Payload serialised into ServiceBusMessage, sending now",
                )

                sender.send_messages(single_message)
                _diag_log(
                    verbose_diagnostics,
                    f"Send completed successfully in {time.time() - send_start:.3f}s",
                )
                logging.info("Sending data to the dashboard .. success")
                return 1
    except Exception as e:
        _diag_log(
            verbose_diagnostics,
            f"Send raised exception for queue={config.queue_name}, "
            f"endpoint={endpoint}, context={context}: {e}",
        )
        logging.exception(f"Exception: {e}")
        return 0


def main():
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.DEBUG)

    state = state_utils.init_state()

    state["latitude"] = 56.61
    state["longitude"] = -1.27
    state["copepodCount"] = 24
    state["nonCopepodCount"] = 36
    state["detritusCount"] = 999
    send(state)


if __name__ == "__main__":
    main()
