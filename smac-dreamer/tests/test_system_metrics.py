from smacdreamer import system_metrics


def test_collect_omits_unavailable_main_rss(monkeypatch):
    monkeypatch.setattr(system_metrics, "rss_bytes", lambda pid=None: None)
    metrics = system_metrics.collect_system_metrics()
    assert "system/main_rss_gb" not in metrics
    assert metrics["system/main_rss_available"] == 0.0


def test_collect_worker_rss_when_available(monkeypatch):
    monkeypatch.setattr(system_metrics, "rss_bytes", lambda pid=None: 1024 ** 3)
    metrics = system_metrics.collect_system_metrics(worker_infos=[{"slot": 2, "pid": 123, "generation": 4}])
    assert metrics["system/main_rss_gb"] == 1.0
    assert metrics["system/worker_2_rss_gb"] == 1.0
    assert metrics["system/worker_2_generation"] == 4.0
