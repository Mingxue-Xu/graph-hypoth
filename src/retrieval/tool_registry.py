from __future__ import annotations

from typing import Any

from src.events import stable_hash_payload
from src.retrieval._util import dump as _base_dump
from src.retrieval._util import get_attr_or_key as _get
from src.retrieval.artifacts import write_retrieval_artifacts
from src.retrieval.ledger import EvidenceLedger
from src.retrieval.service import RetrievalService
from src.runtime_trace import record_runtime_event
from src.state import RetrievedEvidence


def _dump(value: Any) -> Any:
    return _base_dump(
        value, tuples=True, str_keys=True, exclude_computed=True, dict_fallback=False
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _looks_like_domain(value: str) -> bool:
    return (
        "." in value
        and " " not in value
        and "/" not in value
        and "://" not in value
    )


def _verification_status_counts(evidence: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in evidence:
        metadata = item.get("metadata", {})
        selection = (
            metadata.get("quote_selection", {})
            if isinstance(metadata, dict)
            else {}
        )
        if not isinstance(selection, dict) or not selection:
            continue
        status = str(selection.get("verification_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
        candidate_status = selection.get("candidate_verification_status")
        if candidate_status:
            key = f"candidate_{candidate_status}"
            counts[key] = counts.get(key, 0) + 1
    return counts


def _cost_dollars(source_statuses: list[dict[str, Any]]) -> list[float]:
    costs: list[float] = []
    for status in source_statuses:
        metadata = status.get("metadata", {})
        if "costDollars" in metadata:
            costs.append(float(metadata["costDollars"]))
    return costs


def build_retrieval_event_payload(
    *,
    result: dict[str, Any],
    called_by: str,
) -> dict[str, Any]:
    """Build the shared SQLite audit payload for any host-side paper search."""
    evidence = result.get("evidence", [])
    source_statuses = result.get("source_statuses", [])
    payload = {
        "tool_name": "search_papers",
        "tool_call_id": result["tool_call_id"],
        "called_by": called_by,
        "query": result["query"],
        "filters": result.get("filters", {}),
        "sources": result.get("sources", []),
        "source_queries": {
            status["source"]: status.get("source_query")
            for status in source_statuses
        },
        "input_hash": result.get("input_hash", ""),
        "retrieval_config_hash": result.get("retrieval_config_hash", ""),
        "result_count": len(evidence),
        "evidence_ids": [
            item.get("evidence_id") for item in evidence if item.get("evidence_id")
        ],
        "evidence_hashes": [stable_hash_payload(item) for item in evidence],
        "source_statuses": source_statuses,
        "warnings": result.get("warnings", []),
        "errors": result.get("errors", []),
        "elapsed_ms": result.get("elapsed_ms", 0),
        "cost_dollars": _cost_dollars(source_statuses),
        "verification_status_counts": _verification_status_counts(evidence),
        "evidence": evidence,
    }
    if result.get("retrieval_artifacts") is not None:
        payload["retrieval_artifacts"] = result["retrieval_artifacts"]
    return payload


class RetrievalToolRegistry:
    def __init__(
        self,
        *,
        run_id: str,
        thread_id: str,
        checkpoint_id: str,
        round_index: int,
        config: Any,
        service: RetrievalService,
        ledger: EvidenceLedger | None = None,
        initial_tool_index: int = 1,
    ) -> None:
        self.run_id = run_id
        self.thread_id = thread_id
        self.checkpoint_id = checkpoint_id
        self.round_index = round_index
        self.config = config
        self.service = service
        self.ledger = ledger
        self._next_tool_index = initial_tool_index
        self._calls_per_run = 0
        self._calls_by_agent_round: dict[tuple[str, int], int] = {}
        self._pending_event_payloads: list[dict[str, Any]] = []

    def search_papers(
        self,
        query: str,
        sources: list[str] | None = None,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
        called_by: str = "agent",
    ) -> dict[str, Any]:
        tool_call_id = self._assign_tool_call_id()
        record_runtime_event(
            "agent_tool_call",
            run_id=self.run_id,
            thread_id=self.thread_id,
            actor=called_by,
            target="search_papers",
            direction="request",
            payload={
                "checkpoint_id": self.checkpoint_id,
                "round_index": self.round_index,
                "tool_call_id": tool_call_id,
                "query": query,
                "sources": sources,
                "limit": limit,
                "filters": filters or {},
            },
        )
        sources, filters, input_warnings = self._normalize_tool_inputs(
            sources=sources,
            filters=filters,
        )
        if self._budget_exceeded(called_by):
            result = self._budget_result(
                tool_call_id=tool_call_id,
                query=query,
                sources=sources or [],
                limit=limit,
                filters=filters or {},
                warnings=input_warnings,
                called_by=called_by,
            )
            self._pending_event_payloads.append(
                self._event_payload(result=result, called_by=called_by)
            )
            record_runtime_event(
                "agent_tool_call",
                run_id=self.run_id,
                thread_id=self.thread_id,
                actor=called_by,
                target="search_papers",
                direction="response",
                payload=result,
            )
            return result

        self._record_budget_use(called_by)
        try:
            search_result = self.service.search_papers(
                run_id=self.run_id,
                query=query,
                sources=sources,
                limit=limit,
                filters=filters,
                called_by=called_by,
                tool_call_id=tool_call_id,
            )
        except Exception as exc:
            result_payload = self._tool_error_result(
                tool_call_id=tool_call_id,
                query=query,
                sources=sources or [],
                limit=limit,
                filters=filters or {},
                warnings=input_warnings,
                error=f"search_papers failed: {exc}",
            )
        else:
            result_payload = self.service.result_to_dict(search_result)
            if input_warnings:
                result_payload = dict(result_payload)
                result_payload["warnings"] = [
                    *input_warnings,
                    *result_payload.get("warnings", []),
                ]
        result_payload = self._admit_evidence(result_payload, called_by=called_by)
        result_payload = self._attach_retrieval_artifacts(result_payload)
        self._pending_event_payloads.append(
            self._event_payload(
                result=result_payload,
                called_by=called_by,
            )
        )
        record_runtime_event(
            "agent_tool_call",
            run_id=self.run_id,
            thread_id=self.thread_id,
            actor=called_by,
            target="search_papers",
            direction="response",
            payload=result_payload,
        )
        return result_payload

    def drain_pending_event_payloads(self) -> list[dict[str, Any]]:
        payloads = list(self._pending_event_payloads)
        self._pending_event_payloads.clear()
        return payloads

    def _assign_tool_call_id(self) -> str:
        tool_call_id = f"tool_{self._next_tool_index:06d}"
        self._next_tool_index += 1
        return tool_call_id

    def _budget_exceeded(self, called_by: str) -> bool:
        budget = _get(self.config, "tool_budget")
        max_calls_per_run = _optional_int(_get(budget, "max_calls_per_run"))
        max_calls_per_agent_round = _optional_int(
            _get(budget, "max_calls_per_agent_round")
        )
        if (
            max_calls_per_run is not None
            and self._calls_per_run >= max_calls_per_run
        ):
            return True
        key = self._agent_round_key(called_by)
        return max_calls_per_agent_round is not None and (
            self._calls_by_agent_round.get(key, 0)
            >= max_calls_per_agent_round
        )

    def _budget_warning(self, called_by: str) -> str:
        """A truncation warning that names the budget cap(s) at their limit, so a caller
        sees retrieval was skipped (not a silent empty) and which limit to raise."""
        budget = _get(self.config, "tool_budget")
        run_cap = _optional_int(_get(budget, "max_calls_per_run"))
        round_cap = _optional_int(_get(budget, "max_calls_per_agent_round"))
        tripped: list[str] = []
        if run_cap is not None and self._calls_per_run >= run_cap:
            tripped.append(f"max_calls_per_run={run_cap}")
        round_key = self._agent_round_key(called_by)
        if round_cap is not None and self._calls_by_agent_round.get(round_key, 0) >= round_cap:
            tripped.append(f"max_calls_per_agent_round={round_cap}")
        if not tripped:
            return "retrieval tool budget exceeded"
        # Keep the "retrieval tool budget exceeded" substring stable — callers classify
        # budget-truncation warnings by this prefix rather than by the detail suffix.
        return f"retrieval tool budget exceeded ({', '.join(tripped)}); query skipped"

    def _record_budget_use(self, called_by: str) -> None:
        self._calls_per_run += 1
        key = self._agent_round_key(called_by)
        self._calls_by_agent_round[key] = (
            self._calls_by_agent_round.get(key, 0) + 1
        )

    def _agent_round_key(self, called_by: str) -> tuple[str, int]:
        return (called_by, self.round_index)

    def _normalize_tool_inputs(
        self,
        *,
        sources: list[str] | None,
        filters: dict[str, Any] | None,
    ) -> tuple[list[str] | None, dict[str, Any], list[str]]:
        dumped_filters = _dump(filters) if filters is not None else {}
        filter_payload = dict(dumped_filters) if isinstance(dumped_filters, dict) else {}
        warnings: list[str] = []
        normalized_sources = self._normalize_tool_sources(
            sources=sources,
            filters=filter_payload,
            warnings=warnings,
        )
        self._normalize_text_filter(
            filters=filter_payload,
            field_name="include_text",
            warnings=warnings,
        )
        self._normalize_text_filter(
            filters=filter_payload,
            field_name="exclude_text",
            warnings=warnings,
        )
        return normalized_sources, filter_payload, warnings

    def _normalize_tool_sources(
        self,
        *,
        sources: list[str] | None,
        filters: dict[str, Any],
        warnings: list[str],
    ) -> list[str] | None:
        if sources is None:
            return None

        available_sources = self._available_source_names()
        configured_sources = self._configured_source_names(available_sources)
        normalized_sources: list[str] = []
        domain_filters: list[str] = []
        ignored_sources: list[str] = []
        for source in sources:
            source_name = str(source).strip().lower()
            if not source_name:
                continue
            if source_name in available_sources:
                normalized_sources.append(source_name)
            elif _looks_like_domain(source_name):
                domain_filters.append(source_name)
            else:
                ignored_sources.append(source_name)

        if domain_filters:
            existing_domains = filters.get("include_domains")
            existing_domain_filters = (
                [
                    str(item).strip().lower()
                    for item in existing_domains
                    if str(item).strip()
                ]
                if isinstance(existing_domains, list)
                else []
            )
            filters["include_domains"] = _dedupe_strings(
                [*existing_domain_filters, *domain_filters]
            )
            warnings.extend(
                [
                    f"search_papers treated unsupported source '{domain_filter}' "
                    "as include_domains filter"
                    for domain_filter in domain_filters
                ]
            )
            if (
                "exa" in available_sources
                and "exa" in configured_sources
                and "exa" not in normalized_sources
            ):
                normalized_sources.append("exa")

        if ignored_sources:
            warnings.extend(
                [
                    f"search_papers ignored unsupported source '{source}'"
                    for source in ignored_sources
                ]
            )

        normalized_sources = _dedupe_strings(normalized_sources)
        return normalized_sources or None

    def _available_source_names(self) -> set[str]:
        service_sources = getattr(self.service, "sources", None)
        if isinstance(service_sources, dict):
            return {str(source) for source in service_sources}
        if isinstance(service_sources, list | tuple | set):
            return {str(source) for source in service_sources}
        return {"arxiv", "exa", "fake"}

    def _configured_source_names(self, available_sources: set[str]) -> set[str]:
        selector = _get(self.config, "selected_source_names")
        if callable(selector):
            try:
                return {str(source) for source in selector()}
            except Exception:
                return set(available_sources)
        sources = _get(self.config, "sources")
        optional_sources = _get(self.config, "optional_sources")
        configured: set[str] = set()
        if isinstance(sources, list | tuple | set):
            configured.update(str(source) for source in sources)
        if isinstance(optional_sources, list | tuple | set):
            configured.update(str(source) for source in optional_sources)
        return configured or set(available_sources)

    def _normalize_text_filter(
        self,
        *,
        filters: dict[str, Any],
        field_name: str,
        warnings: list[str],
    ) -> None:
        raw_value = filters.get(field_name)
        if raw_value is None:
            return
        if isinstance(raw_value, str):
            values = [raw_value]
        elif isinstance(raw_value, list | tuple):
            values = [str(item) for item in raw_value]
        else:
            filters.pop(field_name, None)
            warnings.append(
                f"search_papers ignored {field_name} because it must be a string array"
            )
            return

        values = [" ".join(value.split()) for value in values if value.strip()]
        if not values:
            filters.pop(field_name, None)
            return
        if len(values) > 1:
            filters.pop(field_name, None)
            warnings.append(
                f"search_papers ignored {field_name} "
                "because it accepts at most one string"
            )
            return
        if len(values[0].split()) > 5:
            filters.pop(field_name, None)
            warnings.append(
                f"search_papers ignored {field_name} "
                "because the string must be at most five words"
            )
            return
        filters[field_name] = values

    def _admit_evidence(self, result: dict[str, Any], *, called_by: str) -> dict[str, Any]:
        if self.ledger is None:
            return result
        incoming = [
            RetrievedEvidence.model_validate(item)
            for item in result.get("evidence", [])
        ]
        admitted = self.ledger.add_retrieved(
            items=incoming,
            query=result["query"],
            retrieved_by=called_by,
            tool_call_id=result["tool_call_id"],
        )
        result = dict(result)
        result["evidence"] = [item.model_dump(mode="json") for item in admitted]
        return result

    def _budget_result(
        self,
        *,
        tool_call_id: str,
        query: str,
        sources: list[str],
        limit: int | None,
        filters: dict[str, Any],
        warnings: list[str] | None = None,
        called_by: str = "agent",
    ) -> dict[str, Any]:
        retrieval_config_hash = stable_hash_payload(_dump(self.config))
        input_hash = stable_hash_payload(
            {
                "tool_name": "search_papers",
                "query": query,
                "sources": sources,
                "limit": limit,
                "filters": filters,
                "retrieval_config_hash": retrieval_config_hash,
            }
        )
        return {
            "tool_call_id": tool_call_id,
            "query": query,
            "filters": filters,
            "sources": sources,
            "evidence": [],
            "source_statuses": [],
            "warnings": [*(warnings or []), self._budget_warning(called_by)],
            "errors": [],
            "elapsed_ms": 0,
            "input_hash": input_hash,
            "retrieval_config_hash": retrieval_config_hash,
        }

    def _tool_error_result(
        self,
        *,
        tool_call_id: str,
        query: str,
        sources: list[str],
        limit: int | None,
        filters: dict[str, Any],
        warnings: list[str],
        error: str,
    ) -> dict[str, Any]:
        retrieval_config_hash = stable_hash_payload(_dump(self.config))
        input_hash = stable_hash_payload(
            {
                "tool_name": "search_papers",
                "query": query,
                "sources": sources,
                "limit": limit,
                "filters": filters,
                "retrieval_config_hash": retrieval_config_hash,
            }
        )
        return {
            "tool_call_id": tool_call_id,
            "query": query,
            "filters": filters,
            "sources": sources,
            "evidence": [],
            "source_statuses": [],
            "warnings": warnings,
            "errors": [error],
            "elapsed_ms": 0,
            "input_hash": input_hash,
            "retrieval_config_hash": retrieval_config_hash,
        }

    def _event_payload(
        self,
        *,
        result: dict[str, Any],
        called_by: str,
    ) -> dict[str, Any]:
        return build_retrieval_event_payload(result=result, called_by=called_by)

    def _attach_retrieval_artifacts(self, result: dict[str, Any]) -> dict[str, Any]:
        artifact_config = _get(self.config, "artifacts")
        if not bool(_get(artifact_config, "enabled", True)):
            return result
        paths = write_retrieval_artifacts(
            run_id=self.run_id,
            tool_call_id=result["tool_call_id"],
            source="search_papers",
            payload=result,
            write_json=bool(_get(artifact_config, "write_json", True)),
            write_html=bool(_get(artifact_config, "write_html", True)),
        )
        if paths is None:
            return result
        result = dict(result)
        result["retrieval_artifacts"] = paths.model_dump()
        return result

def build_retrieval_stack(
    retrieval_config: Any,
    log_store: Any,
    *,
    run_id: str,
    thread_id: str,
    checkpoint_id: str,
    initial_tool_index: int = 1,
    service: RetrievalService | None = None,
    existing: list[RetrievedEvidence] | None = None,
) -> tuple[RetrievalService, EvidenceLedger, RetrievalToolRegistry]:
    """Assemble the RetrievalService/EvidenceLedger/RetrievalToolRegistry trio every
    retrieval call site builds the same way. ``service`` lets a caller share one
    already-built RetrievalService across several stacks instead of constructing a new
    one; ``existing`` seeds the ledger with evidence already retrieved this run."""
    if service is None:
        service = RetrievalService(config=retrieval_config, log_store=log_store)
    ledger = EvidenceLedger(
        trust_policy=retrieval_config.trust_policy,
        trust_tier_priority=retrieval_config.ranking.trust_tier_priority,
        source_order=retrieval_config.ranking.source_order,
        existing=existing,
    )
    registry = RetrievalToolRegistry(
        run_id=run_id,
        thread_id=thread_id,
        checkpoint_id=checkpoint_id,
        round_index=0,
        config=retrieval_config,
        service=service,
        ledger=ledger,
        initial_tool_index=initial_tool_index,
    )
    return service, ledger, registry
