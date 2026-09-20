"""
Phase 3 Test Suite — RAG Knowledge Base
=========================================
Validates semantic search quality over industrial engineering documents.

Requires the RAG index to be built first:
    python scripts/build_rag_index.py

Run with:
    pytest tests/test_rag_tools.py -v
"""

import pytest
from agent.tools.rag_tools import (
    query_knowledge_base,
    query_tema_standards,
    query_vibration_standards,
    query_bearing_failures,
    query_maintenance_sop,
    get_knowledge_base_status,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def kb_status():
    return get_knowledge_base_status()


# ===========================================================================
# 1. Knowledge Base Status
# ===========================================================================

class TestKnowledgeBaseStatus:

    def test_index_is_ready(self, kb_status):
        assert kb_status["is_ready"] is True, \
            f"RAG index not ready. Run: python scripts/build_rag_index.py\nError: {kb_status.get('error', kb_status.get('message'))}"

    def test_expected_sources_present(self, kb_status):
        expected = {"TEMA_standards", "ISO_10816_vibration_standard",
                    "bearing_failure_catalog", "maintenance_sop"}
        actual   = set(kb_status.get("sources", []))
        missing  = expected - actual
        assert not missing, f"Missing sources in index: {missing}"

    def test_sufficient_chunks(self, kb_status):
        """Should have at least 50 chunks across 4 documents."""
        assert kb_status["n_chunks"] >= 50, \
            f"Only {kb_status['n_chunks']} chunks — expected >= 50"

    def test_all_four_sources_indexed(self, kb_status):
        assert len(kb_status.get("sources", [])) == 4


# ===========================================================================
# 2. General Knowledge Base Query
# ===========================================================================

class TestQueryKnowledgeBase:

    def test_query_returns_results(self):
        result = query_knowledge_base("fouling resistance crude oil TEMA")
        assert result["n_results"] > 0

    def test_query_has_required_keys(self):
        result   = query_knowledge_base("bearing vibration zone")
        required = {"query", "n_results", "results", "top_answer"}
        assert not (required - set(result.keys()))

    def test_results_have_required_fields(self):
        result = query_knowledge_base("ISO 10816 zone classification")
        for r in result["results"]:
            required = {"rank", "text", "source", "distance", "relevance_score"}
            assert not (required - set(r.keys())), f"Missing keys in result: {r.keys()}"

    def test_results_ranked_by_relevance(self):
        """Results must be ordered by ascending distance (most relevant first)."""
        result    = query_knowledge_base("cleaning heat exchanger fouling")
        distances = [r["distance"] for r in result["results"]]
        assert distances == sorted(distances), "Results not ordered by distance"

    def test_top_answer_is_string(self):
        result = query_knowledge_base("TEMA cleaning methods")
        assert isinstance(result["top_answer"], str)
        assert len(result["top_answer"]) > 20

    def test_n_results_parameter_respected(self):
        result = query_knowledge_base("bearing failure", n_results=2)
        assert result["n_results"] <= 2

    def test_source_filter_works(self):
        result = query_knowledge_base(
            "fouling resistance",
            n_results=5,
            source_filter="TEMA_standards",
        )
        for r in result["results"]:
            assert r["source"] == "TEMA_standards", \
                f"Got result from unexpected source: {r['source']}"

    def test_invalid_source_filter_returns_empty_or_valid(self):
        """Filtering by non-existent source should return 0 results (not crash)."""
        result = query_knowledge_base("anything", source_filter="nonexistent_source")
        assert "results" in result   # should not raise


# ===========================================================================
# 3. Semantic Relevance Quality Tests
# ===========================================================================

class TestSemanticRelevance:

    def test_tema_query_returns_tema_source(self):
        """Question about fouling resistance should retrieve TEMA docs."""
        result   = query_knowledge_base("maximum fouling resistance allowed for crude oil exchangers")
        top_sources = [r["source"] for r in result["results"][:3]]
        assert "TEMA_standards" in top_sources, \
            f"TEMA_standards not in top 3 results: {top_sources}"

    def test_iso_query_returns_vibration_doc(self):
        """Question about vibration severity zones should retrieve ISO doc."""
        result      = query_knowledge_base("what is ISO Zone D vibration and what action to take")
        top_sources = [r["source"] for r in result["results"][:3]]
        assert "ISO_10816_vibration_standard" in top_sources, \
            f"ISO doc not in top 3: {top_sources}"

    def test_bearing_query_returns_failure_catalog(self):
        """Question about outer race spalling should retrieve bearing failure catalog."""
        result      = query_knowledge_base("outer race spalling symptoms and vibration signature")
        top_sources = [r["source"] for r in result["results"][:3]]
        assert "bearing_failure_catalog" in top_sources, \
            f"Failure catalog not in top 3: {top_sources}"

    def test_sop_query_returns_maintenance_doc(self):
        """Question about cleaning procedure should retrieve SOP."""
        result      = query_knowledge_base("how to chemically clean a heat exchanger with asphaltene deposits")
        top_sources = [r["source"] for r in result["results"][:3]]
        assert "maintenance_sop" in top_sources, \
            f"Maintenance SOP not in top 3: {top_sources}"

    def test_rf_limit_value_in_answer(self):
        """TEMA fouling resistance answer must contain the limit value (0.0005)."""
        result = query_knowledge_base("TEMA fouling resistance limit crude oil m2 K/W")
        combined = " ".join(r["text"] for r in result["results"])
        assert "0.0005" in combined or "0.00050" in combined, \
            "TEMA Rf limit (0.0005) not found in retrieved passages"

    def test_iso_zone_d_trigger_in_answer(self):
        """ISO Zone D query should retrieve the 7.1 mm/s threshold."""
        result   = query_knowledge_base("ISO 10816 danger zone threshold mm/s")
        combined = " ".join(r["text"] for r in result["results"])
        assert "7.1" in combined, "ISO Zone D value (7.1 mm/s) not found in retrieved passages"


# ===========================================================================
# 4. Domain-Specific Shortcut Functions
# ===========================================================================

class TestDomainShortcuts:

    def test_query_tema_standards(self):
        result = query_tema_standards("fouling resistance cleaning interval")
        assert result["n_results"] > 0
        for r in result["results"]:
            assert r["source"] == "TEMA_standards"

    def test_query_vibration_standards(self):
        result = query_vibration_standards("Zone B acceptable vibration")
        assert result["n_results"] > 0
        for r in result["results"]:
            assert r["source"] == "ISO_10816_vibration_standard"

    def test_query_bearing_failures(self):
        result = query_bearing_failures("IMS Test 2 Bearing 1 failure")
        assert result["n_results"] > 0
        for r in result["results"]:
            assert r["source"] == "bearing_failure_catalog"

    def test_query_maintenance_sop(self):
        result = query_maintenance_sop("hydroblasting procedure safety prerequisites")
        assert result["n_results"] > 0
        for r in result["results"]:
            assert r["source"] == "maintenance_sop"

    def test_shortcuts_json_serializable(self):
        import json
        result = query_tema_standards("fouling")
        json.dumps(result)
