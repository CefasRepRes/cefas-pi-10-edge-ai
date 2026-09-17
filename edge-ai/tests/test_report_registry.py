"""Unit tests for the window-based report registry logic introduced in edge-ai.py.

These tests validate the time-matching, expiry, and drain-queue behaviours
without importing the full edge-ai module (which requires a live camera
environment, ipconfig.txt, GPU libraries, etc.).

Run with:
    pytest edge-ai/tests/test_report_registry.py -v
"""

import queue
import threading
import time

import pytest

# ---------------------------------------------------------------------------
# Minimal replica of the apply_hitsmisses_correction logic for isolated testing
# ---------------------------------------------------------------------------

LABEL_GROUPS = {"Detritus": ["Detritus"], "Copepod": ["Copepod"]}
REPORT_INTERVAL = 60  # seconds


def _apply_hitsmisses_correction(packet, hit, miss):
    """Replica of apply_hitsmisses_correction from edge-ai.py."""
    if hit > 0:
        packet["hits"] = hit
        packet["misses"] = miss
        tot = 0
        for label in LABEL_GROUPS.keys():
            packet[f"{label}Count"] = round(
                ((hit + miss) / hit)
                * packet["edgeSubRate"]
                * packet[f"uncorrected_{label}Count"]
                / 34,
                3,
            )
            tot += packet["edgeSubRate"] * packet[f"uncorrected_{label}Count"]
        packet["totalCount"] = tot


def _make_report_packet(window_end, status="pending", edge_sub_rate=1):
    """Build a packet whose keys match LABEL_GROUPS exactly."""
    p = {
        "window_start": window_end - REPORT_INTERVAL,
        "window_end": window_end,
        "status": status,
        "edgeSubRate": edge_sub_rate,
        "totalCount": 0,
        "hits": 0,
        "misses": 0,
    }
    for label in LABEL_GROUPS.keys():
        p[f"uncorrected_{label}Count"] = 10
        p[f"{label}Count"] = 0
    return p


# ---------------------------------------------------------------------------
# Replica of _expire_pending_packets for isolated testing
# ---------------------------------------------------------------------------


def _expire_pending_packets(registry, last_hm_period_end, report_interval):
    """Replica of the inner _expire_pending_packets from edge-ai.py."""
    if last_hm_period_end is None:
        return []
    hm_window_start = last_hm_period_end - 10 * report_interval
    to_remove = [
        p
        for p in registry
        if p["status"] == "pending" and p["window_end"] < hm_window_start
    ]
    expired = []
    for packet in to_remove:
        expired.append(packet)
        registry.remove(packet)
    return expired


# ---------------------------------------------------------------------------
# Replica of _drain_image_queue for isolated testing
# ---------------------------------------------------------------------------


def _drain_image_queue(mq):
    """Replica of _drain_image_queue from edge-ai.py."""
    kept = []
    while True:
        try:
            item = mq.get_nowait()
            _, _img, _hm = item
            if _hm is not None:
                kept.append(item)
        except queue.Empty:
            break
    for item in kept:
        try:
            mq.put_nowait(item)
        except queue.Full:
            pass  # best-effort


# ---------------------------------------------------------------------------
# Replica of HitsMisses time-matching logic for isolated testing
# ---------------------------------------------------------------------------


def _match_hitsmisses(registry, rows, period_end, report_interval):
    """Match HitsMisses rows to pending registry packets by time."""
    matched = []
    for i, (hit, miss) in enumerate(rows):
        target_window_end = period_end - (9 - i) * report_interval
        best_match = None
        best_diff = float("inf")
        for packet in registry:
            if packet["status"] == "pending":
                diff = abs(packet["window_end"] - target_window_end)
                if diff < report_interval / 2 and diff < best_diff:
                    # Allow up to half a report interval of drift.
                    best_diff = diff
                    best_match = packet
        if best_match is not None and hit > 0:
            _apply_hitsmisses_correction(best_match, hit, miss)
            best_match["status"] = "validated"
            matched.append(best_match)
    return matched


# ===========================================================================
# Tests: apply_hitsmisses_correction
# ===========================================================================


class TestApplyHitsmissesCorrection:
    def test_correction_applied_when_hit_positive(self):
        packet = _make_report_packet(window_end=1000.0)
        _apply_hitsmisses_correction(packet, hit=8, miss=2)
        assert packet["hits"] == 8
        assert packet["misses"] == 2
        # formula: ((hit + miss) / hit) * edgeSubRate * uncorrected_count / 34
        assert packet["CopepodCount"] == pytest.approx(((8 + 2) / 8) * 1 * 10 / 34, rel=1e-3)

    def test_no_change_when_hit_is_zero(self):
        packet = _make_report_packet(window_end=1000.0)
        original_count = packet["CopepodCount"]
        _apply_hitsmisses_correction(packet, hit=0, miss=5)
        assert packet["CopepodCount"] == original_count

    def test_total_count_updated(self):
        packet = _make_report_packet(window_end=1000.0, edge_sub_rate=2)
        _apply_hitsmisses_correction(packet, hit=10, miss=0)
        # totalCount = sum(edgeSubRate * uncorrected_count) for all labels
        expected_total = 2 * 10 + 2 * 10  # Detritus + Copepod, each 10 uncorrected
        assert packet["totalCount"] == expected_total


# ===========================================================================
# Tests: _expire_pending_packets
# ===========================================================================


class TestExpirePendingPackets:
    def test_no_expiry_when_no_hitsmisses_received(self):
        registry = [_make_report_packet(window_end=1000.0)]
        expired = _expire_pending_packets(registry, None, REPORT_INTERVAL)
        assert expired == []
        assert len(registry) == 1

    def test_packet_inside_window_not_expired(self):
        T = 1200.0
        # packet ending at T - 5*60 is inside the 10-minute window
        registry = [_make_report_packet(window_end=T - 5 * REPORT_INTERVAL)]
        expired = _expire_pending_packets(registry, T, REPORT_INTERVAL)
        assert expired == []
        assert len(registry) == 1

    def test_packet_outside_window_is_expired(self):
        T = 1200.0
        # packet ending at T - 11*60 is before the window start (T - 10*60)
        registry = [_make_report_packet(window_end=T - 11 * REPORT_INTERVAL)]
        expired = _expire_pending_packets(registry, T, REPORT_INTERVAL)
        assert len(expired) == 1
        assert len(registry) == 0

    def test_validated_packet_is_never_expired(self):
        T = 1200.0
        # Even an old validated packet must NOT be expired
        registry = [_make_report_packet(window_end=T - 11 * REPORT_INTERVAL, status="validated")]
        expired = _expire_pending_packets(registry, T, REPORT_INTERVAL)
        assert expired == []
        assert len(registry) == 1

    def test_only_pending_packets_outside_window_removed(self):
        T = 2000.0
        old_pending = _make_report_packet(window_end=T - 12 * REPORT_INTERVAL)
        old_validated = _make_report_packet(window_end=T - 12 * REPORT_INTERVAL, status="validated")
        new_pending = _make_report_packet(window_end=T - 3 * REPORT_INTERVAL)
        registry = [old_pending, old_validated, new_pending]
        expired = _expire_pending_packets(registry, T, REPORT_INTERVAL)
        assert len(expired) == 1
        assert expired[0] is old_pending
        assert len(registry) == 2
        assert old_validated in registry
        assert new_pending in registry


# ===========================================================================
# Tests: _drain_image_queue
# ===========================================================================


class TestDrainImageQueue:
    def test_drain_removes_image_items(self):
        mq = queue.Queue(50)
        mq.put(("img1.tif", b"bytes", None))
        mq.put(("img2.tif", b"bytes", None))
        _drain_image_queue(mq)
        assert mq.empty()

    def test_drain_keeps_hitsmisses_items(self):
        mq = queue.Queue(50)
        mq.put(("img1.tif", b"bytes", None))
        mq.put(("HitsMisses.txt", None, "1858,0\n2101,0\n"))
        mq.put(("img2.tif", b"bytes", None))
        _drain_image_queue(mq)
        assert mq.qsize() == 1
        fname, img, hm = mq.get_nowait()
        assert fname == "HitsMisses.txt"
        assert hm == "1858,0\n2101,0\n"

    def test_drain_empty_queue_is_safe(self):
        mq = queue.Queue(50)
        _drain_image_queue(mq)  # must not raise

    def test_drain_preserves_multiple_hitsmisses_items(self):
        mq = queue.Queue(50)
        mq.put(("HitsMisses.txt", None, "batch1"))
        mq.put(("img.tif", b"data", None))
        mq.put(("HitsMisses.txt", None, "batch2"))
        _drain_image_queue(mq)
        remaining = []
        while not mq.empty():
            remaining.append(mq.get_nowait())
        assert len(remaining) == 2
        assert all(item[2] is not None for item in remaining)


# ===========================================================================
# Tests: HitsMisses time-matching
# ===========================================================================


class TestHitsMissesMatching:
    def _make_10_packets(self, period_end):
        """Build 10 pending packets matching the 10 windows before period_end."""
        registry = []
        for i in range(10):
            # Row i (0-oldest) should match window ending at period_end - (9-i)*60
            window_end = period_end - (9 - i) * REPORT_INTERVAL
            registry.append(_make_report_packet(window_end=window_end))
        return registry

    def test_all_rows_matched_exactly(self):
        T = 2000.0
        registry = self._make_10_packets(T)
        rows = [(100 + i, 0) for i in range(10)]
        matched = _match_hitsmisses(registry, rows, T, REPORT_INTERVAL)
        assert len(matched) == 10
        assert all(p["status"] == "validated" for p in registry)

    def test_no_match_when_registry_empty(self):
        T = 2000.0
        rows = [(100, 0)] * 10
        matched = _match_hitsmisses([], rows, T, REPORT_INTERVAL)
        assert matched == []

    def test_row_with_hit_zero_does_not_change_packet(self):
        T = 2000.0
        registry = self._make_10_packets(T)
        rows = [(0, 5) if i == 5 else (100, 0) for i in range(10)]
        _match_hitsmisses(registry, rows, T, REPORT_INTERVAL)
        # Packet corresponding to row 5 (hit=0) should NOT be validated
        target_window_end = T - (9 - 5) * REPORT_INTERVAL
        unmatched = [p for p in registry if abs(p["window_end"] - target_window_end) < 1]
        assert len(unmatched) == 1
        assert unmatched[0]["status"] == "pending"

    def test_late_hitsmisses_matches_within_tolerance(self):
        """HitsMisses arriving slightly late (≤29s) should still match."""
        T_actual = 2000.0  # true period end
        T_arrived = T_actual + 25  # arrived 25 seconds late
        registry = self._make_10_packets(T_actual)
        rows = [(200, 0)] * 10
        matched = _match_hitsmisses(registry, rows, T_arrived, REPORT_INTERVAL)
        # All rows land within 25s of their target; tolerance is 30s, so all match
        assert len(matched) == 10

    def test_already_validated_packet_not_matched_again(self):
        T = 2000.0
        registry = self._make_10_packets(T)
        # Pre-validate the first packet
        registry[0]["status"] = "validated"
        rows = [(100, 0)] * 10
        matched = _match_hitsmisses(registry, rows, T, REPORT_INTERVAL)
        # First packet is already validated; only 9 should be newly matched
        assert len(matched) == 9
