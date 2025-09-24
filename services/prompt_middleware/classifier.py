#!/usr/bin/env python3
"""
Minimal intent classifier stub for shadow prompts.
Rule-based mapping from text to intent labels; to be replaced by trained model outputs.
"""

from typing import Dict
import re


INTENT_KEYWORDS = {
    "frontend_perf_render_thrash": ["ui", "render", "scroll", "typing", "lag", "layout", "breathes"],
    "move_work_off_hot_path": ["p95", "hot path", "synchronously", "request path", "spiking"],
    "enforce_api_contracts": ["contract", "downstream api", "feature flags", "region"],
    "instrument_contention_before_tuning": ["contention", "index", "write pattern", "skew", "peak"],
    "separate_user_errors_from_system_faults": ["recoverable", "system faults", "guaranteed", "ops"],
    "revisit_cache_boundary_and_policy": ["cache", "expiration", "coalesce", "precompute", "burst"],
    "add_checkpoints_and_idempotent_retries": ["checkpoint", "retry", "resume", "temp state"],
    "tighten_untrusted_config_parsing": ["untrusted", "config", "loader", "deployment"],
    "introduce_time_scheduler_abstractions": ["concurrency", "time", "scheduling", "reliably"],
    "resource_guardrails_for_backfills": ["backfill", "off-hours", "starve", "interactive"],
    "isolate_global_state_incrementally": ["global state", "reordering", "incrementally", "refactor"],
    "observable_backoff_and_circuit_breaker": ["retries", "backoff", "circuit", "maintenance"],
    "reduce_layout_thrash_and_stage_updates": ["layout", "thrash", "main thread", "typing"],
    "wrap_metaprogramming_with_explicit_api": ["metaprogramming", "boilerplate", "contributors", "magic"],
    "schema_compatibility_policy_checks": ["schema", "compatibility", "forward", "back"],
    "align_event_and_processing_time": ["event-time", "processing-time", "skew", "reconciliation"],
    "add_schema_and_lints_for_plugins": ["plugin", "json", "discoverability", "schema", "lints"],
    "prune_noise_and_surface_runbook_signals": ["runbook", "dashboards", "noise", "signals"]
}


def classify_intent(text: str) -> str:
    """Classify intent using keyword heuristics.

    Returns: intent label or 'unknown'.
    """
    tl = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        hits = sum(1 for k in keywords if k in tl)
        if hits >= 2:
            return intent
    return "unknown"


def score_intents(text: str) -> Dict[str, float]:
    """Return simple scores per intent (keyword hit ratios)."""
    tl = text.lower()
    scores: Dict[str, float] = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        hits = sum(1 for k in keywords if k in tl)
        scores[intent] = hits / max(1, len(keywords))
    return scores


