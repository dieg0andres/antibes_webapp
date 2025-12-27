from django.db import models
from django.utils import timezone


class VolatilityTrackedSymbol(models.Model):
    """
    Symbols to include in the volatility pipeline.

    Keep this small and explicit so the pipeline loops only tickers you care about.
    """
    symbol = models.CharField(max_length=16, unique=True)
    is_active = models.BooleanField(default=True)

    # Optional metadata (future-proofing)
    asset_class = models.CharField(max_length=32, blank=True, default="")  # e.g. "ETF", "Equity"
    notes = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.symbol


class VolatilityPipelineRun(models.Model):
    """
    One pipeline execution (e.g., daily_close). A run can produce multiple snapshots
    across symbols and rv_reference variants.
    """
    run_at = models.DateTimeField(default=timezone.now)
    run_label = models.CharField(max_length=32, default="daily_close")  # later: "intraday_1030", etc.

    STATUS_OK = "ok"
    STATUS_PARTIAL = "partial"
    STATUS_ERROR = "error"
    status = models.CharField(
        max_length=16,
        choices=[(STATUS_OK, "ok"), (STATUS_PARTIAL, "partial"), (STATUS_ERROR, "error")],
        default=STATUS_OK,
    )

    notes = models.TextField(blank=True, default="")

    def __str__(self) -> str:
        return f"{self.run_label} @ {self.run_at.isoformat()} ({self.status})"


class VolatilitySnapshot(models.Model):
    """
    One snapshot per (run, symbol, target_dte_days, rv_reference).

    Stores:
    - aha_metrics: the key decision KPIs
    - iv_result: full IVXATMResult serialized
    - rv_latest: the latest row of the RV panel (serialized)
    - raw_paths: file paths to raw Schwab responses (chain, daily, intraday)
    - flags/notes for troubleshooting
    """
    RVREF_PRIMARY20 = "primary20"
    RVREF_YZ20 = "yz20"
    rv_reference = models.CharField(
        max_length=16,
        choices=[(RVREF_PRIMARY20, "primary20"), (RVREF_YZ20, "yz20")],
        default=RVREF_PRIMARY20,
    )

    run = models.ForeignKey(VolatilityPipelineRun, on_delete=models.CASCADE, related_name="snapshots")
    symbol = models.ForeignKey(VolatilityTrackedSymbol, on_delete=models.CASCADE, related_name="snapshots")

    target_dte_days = models.FloatField(default=30.0)

    status = models.CharField(
        max_length=16,
        choices=[("ok", "ok"), ("error", "error")],
        default="ok",
    )

    aha_metrics = models.JSONField(default=dict, blank=True)
    iv_result = models.JSONField(default=dict, blank=True)
    rv_latest = models.JSONField(default=dict, blank=True)

    flags = models.JSONField(default=dict, blank=True)
    notes = models.JSONField(default=list, blank=True)

    # Raw Schwab response file paths (chain + daily + 15m)
    raw_paths = models.JSONField(default=dict, blank=True)

    error_message = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["run", "symbol", "target_dte_days", "rv_reference"],
                name="uniq_vol_snapshot_per_run_symbol_dte_rvref",
            )
        ]

    def __str__(self) -> str:
        return f"{self.symbol.symbol} dte={self.target_dte_days} {self.rv_reference} ({self.status})"


class VolatilitySessionBar(models.Model):
    """
    Long-history charting store: one row per symbol per session timestamp.

    We store the full panel row as JSON to keep the schema flexible and maximize transparency.
    Later, if needed, we can denormalize selected columns for faster querying.
    """
    symbol = models.ForeignKey(VolatilityTrackedSymbol, on_delete=models.CASCADE, related_name="session_bars")
    session_ts = models.DateTimeField()  # session close timestamp UTC
    data = models.JSONField(default=dict)  # one row of the RV panel series

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["symbol", "session_ts"],
                name="uniq_vol_session_bar_symbol_ts",
            )
        ]
        indexes = [
            models.Index(fields=["symbol", "session_ts"]),
        ]

    def __str__(self) -> str:
        return f"{self.symbol.symbol} @ {self.session_ts.isoformat()}"


class ImpliedVolSnapshot(models.Model):
    """
    IV time series storage: one row per run per symbol per target_dte_days.

    This is optional but makes IV history queries easy (no need to parse iv_result JSON).
    """
    run = models.ForeignKey(VolatilityPipelineRun, on_delete=models.CASCADE, related_name="iv_snaps")
    symbol = models.ForeignKey(VolatilityTrackedSymbol, on_delete=models.CASCADE, related_name="iv_snaps")

    target_dte_days = models.FloatField(default=30.0)

    ivx_calc = models.FloatField(null=True, blank=True)
    ivx_vendor = models.FloatField(null=True, blank=True)
    ivx_diff = models.FloatField(null=True, blank=True)

    flags = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["run", "symbol", "target_dte_days"],
                name="uniq_iv_snapshot_per_run_symbol_dte",
            )
        ]
        indexes = [
            models.Index(fields=["symbol", "created_at"]),
        ]
