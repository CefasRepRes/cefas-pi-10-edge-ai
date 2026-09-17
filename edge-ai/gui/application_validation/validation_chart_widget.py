from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QPen, QFontMetrics
from PySide6.QtCore import Qt, QRectF


class ValidationErrorBarChart(QWidget):
    """Dynamic per-class validation chart with Wilson confidence intervals.

    Expects the validation summary produced by ValidationSessionDialog._build_validation_summary(),
    specifically summary["per_class"][class_name]["recall"/"precision"] containing:
      estimate, lower, upper

    The widget redraws whenever set_summary() is called, so the chart updates as more
    images are validated.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.summary = {}
        self.sample_count = 0
        self.setMinimumHeight(280)

    def set_summary(self, summary, sample_count=0):
        self.summary = summary or {}
        self.sample_count = sample_count or 0
        self.update()

    @staticmethod
    def _bounded(value, default=None):
        if value is None:
            return default
        return max(0.0, min(1.0, float(value)))

    def _draw_no_data(self, painter):
        painter.setPen(QPen(QColor("#555555"), 1))
        painter.drawText(
            self.rect(),
            Qt.AlignCenter,
            "Validate images to show per-class precision/recall error bars",
        )

    def _draw_metric_bar(self, painter, *, x, y_base, chart_h, bar_w, metric, colour):
        estimate = self._bounded((metric or {}).get("estimate"))
        if estimate is None:
            return

        lower = self._bounded((metric or {}).get("lower"), estimate)
        upper = self._bounded((metric or {}).get("upper"), estimate)

        bar_h = int(chart_h * estimate)
        painter.fillRect(int(x), int(y_base - bar_h), int(bar_w), int(bar_h), QColor(colour))

        # Wilson interval error bar.
        err_x = int(x + bar_w / 2)
        y_low = int(y_base - chart_h * lower)
        y_high = int(y_base - chart_h * upper)
        cap = max(3, int(bar_w * 0.45))
        painter.setPen(QPen(QColor("#222222"), 2))
        painter.drawLine(err_x, y_high, err_x, y_low)
        painter.drawLine(err_x - cap, y_high, err_x + cap, y_high)
        painter.drawLine(err_x - cap, y_low, err_x + cap, y_low)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("white"))

        per_class = self.summary.get("per_class") or {}
        classes = [cls for cls, stats in sorted(per_class.items(), key=lambda kv: str(kv[0]))]
        classes = [cls for cls in classes if per_class.get(cls)]

        validated = int(self.summary.get("validated_image_count") or 0)
        title = f"Per-class validation error bars ({validated}/{self.sample_count or '?'} images validated)"
        painter.setPen(QPen(QColor("#111111"), 1))
        painter.drawText(10, 18, title)

        if not classes or validated <= 0:
            self._draw_no_data(painter)
            return

        left = 54
        right = 18
        top = 34
        bottom = 76
        chart_w = max(1, self.width() - left - right)
        chart_h = max(1, self.height() - top - bottom)
        y_base = self.height() - bottom

        # Axes and horizontal gridlines.
        painter.setPen(QPen(QColor("#d8d8d8"), 1))
        for tick in (0.0, 0.25, 0.50, 0.75, 1.0):
            y = int(y_base - chart_h * tick)
            painter.drawLine(left, y, left + chart_w, y)
            painter.setPen(QPen(QColor("#555555"), 1))
            painter.drawText(8, y + 4, f"{int(tick * 100)}%")
            painter.setPen(QPen(QColor("#d8d8d8"), 1))

        painter.setPen(QPen(QColor("#222222"), 1))
        painter.drawLine(left, top, left, y_base)
        painter.drawLine(left, y_base, left + chart_w, y_base)

        group_w = chart_w / max(1, len(classes))
        bar_gap = max(2.0, min(5.0, group_w * 0.08))
        bar_w = max(3.0, min(22.0, (group_w - (3 * bar_gap)) / 2.0))
        metrics = QFontMetrics(painter.font())

        recall_colour = "#2F80ED"      # blue
        precision_colour = "#F2994A"   # orange

        for idx, cls in enumerate(classes):
            stats = per_class.get(cls) or {}
            x0 = left + idx * group_w + (group_w - (2 * bar_w + bar_gap)) / 2
            recall_x = x0
            precision_x = x0 + bar_w + bar_gap

            self._draw_metric_bar(
                painter,
                x=recall_x,
                y_base=y_base,
                chart_h=chart_h,
                bar_w=bar_w,
                metric=stats.get("recall") or {},
                colour=recall_colour,
            )
            self._draw_metric_bar(
                painter,
                x=precision_x,
                y_base=y_base,
                chart_h=chart_h,
                bar_w=bar_w,
                metric=stats.get("precision") or {},
                colour=precision_colour,
            )

            # Class label. Rotate when groups are narrow so many classes remain readable.
            label = str(cls)
            available_w = max(20, int(group_w - 4))
            short_label = metrics.elidedText(label, Qt.ElideRight, available_w)
            label_x = left + idx * group_w + group_w / 2
            label_y = y_base + 14
            painter.setPen(QPen(QColor("#222222"), 1))
            if group_w < 72:
                painter.save()
                painter.translate(label_x, label_y + 42)
                painter.rotate(-45)
                painter.drawText(QRectF(-40, -10, 80, 20), Qt.AlignCenter, short_label)
                painter.restore()
            else:
                painter.drawText(
                    QRectF(label_x - group_w / 2, label_y, group_w, 34),
                    Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap,
                    short_label,
                )

        # Legend.
        legend_y = self.height() - 20
        painter.fillRect(60, legend_y - 10, 12, 10, QColor(recall_colour))
        painter.setPen(QPen(QColor("#222222"), 1))
        painter.drawText(78, legend_y, "Recall")
        painter.fillRect(140, legend_y - 10, 12, 10, QColor(precision_colour))
        painter.drawText(158, legend_y, "Precision")
