#! /bin/sh

sleep 10

cd ~/git/rapid-plankton/edge-ai
. env/bin/activate

while true; do
    ./edge-ai.py -c -e --gps -t &
    EDGE_AI_PID=$!
    sleep 24h
    kill $EDGE_AI_PID
    wait $EDGE_AI_PID
done
