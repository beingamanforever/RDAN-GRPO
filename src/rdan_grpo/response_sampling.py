"""Deterministic source coverage for response preflight."""

from __future__ import annotations

from collections.abc import Sequence

PREFLIGHT_SOURCES = ("type1", "type2", "type3", "type4", "rubrichub_instruction_following")


def balanced_preflight_indices(sources: Sequence[str], count: int) -> list[int]:
    """Select a balanced, deterministic spread from every frozen source."""

    if isinstance(count, bool) or not isinstance(count, int) or count < len(PREFLIGHT_SOURCES):
        raise ValueError("preflight count must cover every frozen source")
    pools = {source: [] for source in PREFLIGHT_SOURCES}
    for index, source in enumerate(sources):
        if source not in pools:
            raise ValueError(f"preflight dataset contains unexpected source: {source!r}")
        pools[source].append(index)
    base, remainder = divmod(count, len(PREFLIGHT_SOURCES))
    quotas = {source: base + (offset < remainder) for offset, source in enumerate(PREFLIGHT_SOURCES)}
    if any(len(pools[source]) < quota for source, quota in quotas.items()):
        raise ValueError("preflight dataset lacks enough rows from every frozen source")
    selected = {
        source: [indices[offset * len(indices) // quotas[source]] for offset in range(quotas[source])]
        for source, indices in pools.items()
    }
    return [
        selected[source][offset]
        for offset in range(max(quotas.values()))
        for source in PREFLIGHT_SOURCES
        if offset < quotas[source]
    ]
