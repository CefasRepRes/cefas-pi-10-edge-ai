import importlib.util
import logging
import pathlib
import sys
import types


SENDER_PATH = pathlib.Path(__file__).resolve().parents[1] / "sender.py"


def _load_sender_module(monkeypatch, should_fail=False):
    sent_payloads = []

    class FakeQueueSender:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def send_messages(self, message):
            if should_fail:
                raise RuntimeError("send failed")
            sent_payloads.append(message)

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get_queue_sender(self, _queue_name):
            return FakeQueueSender()

    class FakeServiceBusClient:
        @classmethod
        def from_connection_string(cls, _connstr):
            return FakeClient()

    fake_servicebus = types.ModuleType("azure.servicebus")
    fake_servicebus.ServiceBusClient = FakeServiceBusClient
    fake_servicebus.ServiceBusMessage = lambda payload: payload

    fake_config = types.ModuleType("config")
    fake_config.connstr = (
        "Endpoint=sb://example.servicebus.windows.net/;"
        "SharedAccessKeyName=Test;SharedAccessKey=abc"
    )
    fake_config.queue_name = "rv-dashboard"

    monkeypatch.setitem(sys.modules, "azure.servicebus", fake_servicebus)
    monkeypatch.setitem(sys.modules, "config", fake_config)

    spec = importlib.util.spec_from_file_location("sender_under_test", SENDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, sent_payloads


def test_verbose_send_diagnostics_logging(monkeypatch, caplog):
    sender, sent_payloads = _load_sender_module(monkeypatch)
    caplog.set_level(logging.INFO)

    result = sender.send(
        {"window_start": 123, "status": "validated"},
        verbose_diagnostics=True,
        diagnostics_context={"attempt": 1},
    )

    assert result == 1
    assert len(sent_payloads) == 1
    assert "[send-diagnostics] Preparing payload" in caplog.text
    assert "[send-diagnostics] Send completed successfully" in caplog.text
    assert "attempt" in caplog.text


def test_send_failure_returns_zero_and_logs_diagnostics(monkeypatch, caplog):
    sender, _ = _load_sender_module(monkeypatch, should_fail=True)
    caplog.set_level(logging.INFO)

    result = sender.send({"window_start": 999}, verbose_diagnostics=True)

    assert result == 0
    assert "[send-diagnostics] Send raised exception" in caplog.text
