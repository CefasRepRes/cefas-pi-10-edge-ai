# Implement a simple ring buffer

RING_SIZE = 2048
BUFFER_COUNT = 1024


class RingBuffer:
    def __init__(self):
        # current mapping of fields to filenames
        self.filenames = ["" for x in range(RING_SIZE)]

        self.buffers = [
            ([b""] * BUFFER_COUNT) for x in range(RING_SIZE)
        ]  # data packets that may arrive out of order

        self.counts = [0 for x in range(RING_SIZE)]  # number of packets received

        self.unique_ids = [
            0 for x in range(RING_SIZE)
        ]  # current mapping of field indices to unique ids
