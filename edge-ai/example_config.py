#!/usr/bin/env python3

# ******************************************************************************
# Each PI has a unique serial number
# ******************************************************************************

system_serial_no = "PI10-022.0006"

# ******************************************************************************
# Version stamp the dashboard protocol
# ******************************************************************************

protocol_version = "0.0.3"

# ******************************************************************************
# Add your Service Bus configuration parameters here.
# ******************************************************************************

# example connstr. This is just so you know what to expect. Do not actually pass your connstr via this config.py file, define it as an operating system variable to be secure.
#connstr = "Endpoint=sb://cit-rd-plankton-svcbus.servicebus.windows.net/;SharedAccessKeyName=RVSend;SharedAccessKey=something_like_Debq0m294NwrRu+SyDBeD/REAvzC7JVBD+ASbMTxons="
#queue_name = "rv-dashboard"

# best practice for passing the connstr
import os
connstr = os.environ["SERVICE_BUS_CONNECTION_STR"]
queue_name = os.environ["SERVICE_BUS_QUEUE_NAME"]
