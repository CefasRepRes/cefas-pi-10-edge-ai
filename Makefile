# Run the end-to-end system

run:
	./edge-ai.py --classify --verbose --send --gps --gray --extract --survey "CEND_17_23"

# Run full system with debug logging

debug:
	./edge-ai.py --log-level DEBUG --classify --background-correction --extract --send --gps

# Show help

help:
	./edge-ai.py --help

# Demo the system locally

demo:
	./edge-ai.py --display

clean:
	rm edge-ai.log
	rm *.tif
	rm *.bin
