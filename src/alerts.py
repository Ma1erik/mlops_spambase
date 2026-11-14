import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional


class AlertManager:
    """Менеджер алертов с подавлением повторных срабатываний."""

    def __init__(
        self,
        log_path: str = "reports/LAB8/alerts_sample.jsonl",
        suppression_window_sec: int = 300,
    ):
        self.log_path = log_path
        self.suppression_window_sec = suppression_window_sec
        self._active_alerts: Dict[str, float] = {}
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def trigger(
        self,
        metric: str,
        value: float,
        threshold: float,
        level: str,
        action: str,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Регистрирует алерт, если по показателю нет активного непогашенного алерта.
        Возвращает True, если алерт записан, и False, если подавлен.
        """
        now = timestamp or datetime.now(timezone.utc)
        now_ts = now.timestamp()

        last_ts = self._active_alerts.get(metric)
        if last_ts is not None and (now_ts - last_ts) < self.suppression_window_sec:
            return False

        self._active_alerts[metric] = now_ts
        record = {
            "timestamp": now.isoformat(),
            "metric": metric,
            "value": (
                round(float(value), 4) if isinstance(value, (int, float)) else value
            ),
            "threshold": threshold,
            "level": level,
            "action": action,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True

    def resolve(self, metric: str) -> None:
        """Снимает флаг активного инцидента при нормализации метрики."""
        self._active_alerts.pop(metric, None)

    def reset(self) -> None:
        """Очищает состояние активных алертов."""
        self._active_alerts.clear()
