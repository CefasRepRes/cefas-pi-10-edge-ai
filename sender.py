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


def send(data):
    data = json.dumps(data, indent=4)

    logging.info("Sending data to the dashboard ..")

    try:
        with ServiceBusClient.from_connection_string(config.connstr) as client:
            with client.get_queue_sender(config.queue_name) as sender:
                logging.info(data)

                # Sending a single message
                single_message = ServiceBusMessage(data)

                sender.send_messages(single_message)
                logging.info("Sending data to the dashboard .. success")
                return 1
    except Exception as e:
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
