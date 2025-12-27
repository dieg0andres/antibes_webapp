from django.contrib import admin
from .models import (
    VolatilityTrackedSymbol,
    VolatilityPipelineRun,
    VolatilitySnapshot,
    VolatilitySessionBar,
    ImpliedVolSnapshot,
)

@admin.register(VolatilityTrackedSymbol)
class VolatilityTrackedSymbolAdmin(admin.ModelAdmin):
    list_display = ("symbol", "is_active", "asset_class", "created_at")
    list_filter = ("is_active", "asset_class")
    search_fields = ("symbol",)

@admin.register(VolatilityPipelineRun)
class VolatilityPipelineRunAdmin(admin.ModelAdmin):
    list_display = ("run_at", "run_label", "status")
    list_filter = ("run_label", "status")
    ordering = ("-run_at",)

@admin.register(VolatilitySnapshot)
class VolatilitySnapshotAdmin(admin.ModelAdmin):
    list_display = ("created_at", "run", "symbol", "target_dte_days", "rv_reference", "status")
    list_filter = ("rv_reference", "status", "target_dte_days")
    search_fields = ("symbol__symbol",)
    ordering = ("-created_at",)
    readonly_fields = ("created_at",)

@admin.register(VolatilitySessionBar)
class VolatilitySessionBarAdmin(admin.ModelAdmin):
    list_display = ("symbol", "session_ts", "created_at")
    list_filter = ("symbol",)
    search_fields = ("symbol__symbol",)
    ordering = ("-session_ts",)
    readonly_fields = ("created_at",)

@admin.register(ImpliedVolSnapshot)
class ImpliedVolSnapshotAdmin(admin.ModelAdmin):
    list_display = ("created_at", "run", "symbol", "target_dte_days", "ivx_calc", "ivx_vendor", "ivx_diff")
    list_filter = ("target_dte_days",)
    search_fields = ("symbol__symbol",)
    ordering = ("-created_at",)
    readonly_fields = ("created_at",)
