#!/bin/bash

source env/bin/activate

cleanup() {
    if [ -n "$EDGE_AI_PID" ]; then
        kill "$EDGE_AI_PID" 2>/dev/null
        wait "$EDGE_AI_PID" 2>/dev/null
    fi
    exit 0
}

trap cleanup INT TERM

while true; do
    python edge-ai.py model_18_3classes_RGB_Noushin.pt -b 50 -t --subsampling_rate 10 --sendoneimage &
    EDGE_AI_PID=$!

    sleep 24h

    kill "$EDGE_AI_PID"
    wait "$EDGE_AI_PID"
done

