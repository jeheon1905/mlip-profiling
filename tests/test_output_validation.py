"""
Tests for validating profiling output files.

These tests verify the structure and validity of profiling outputs
without requiring actual model execution. They can be run against
existing output files from previous profiling runs.
"""
import json
import pytest
from pathlib import Path


# =============================================================================
# Schema Definitions
# =============================================================================

SUMMARY_REQUIRED_FIELDS = ["model", "device", "profiler_schedule", "results"]
MODEL_REQUIRED_FIELDS = ["type"]
RESULT_REQUIRED_FIELDS = ["natoms", "qps", "ns_per_day", "timeit_mean_ms", "timeit_std_ms"]
TRACE_REQUIRED_FIELDS = ["traceEvents"]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_summary():
    """Create a valid sample summary for testing."""
    return {
        "model": {
            "type": "mace",
            "path": "/path/to/model.pt",
            "cutoff": 5.0,
            "backend": "e3nn",
        },
        "device": "cuda",
        "profiler_schedule": {
            "wait_steps": 5,
            "warmup_steps": 5,
            "active_steps": 5,
        },
        "timeit_settings": {
            "number": 10,
            "repeat": 5,
        },
        "results": {
            "Cu_fcc_3x3x3_108atoms": {
                "natoms": 108,
                "qps": 30.5,
                "ns_per_day": 2.64,
                "timeit_mean_ms": 32.8,
                "timeit_std_ms": 0.5,
                "operation_times": {
                    "forward": {"gpu_time_ms": 25.0, "cpu_time_ms": 0.1, "count": 1},
                },
            }
        },
    }


@pytest.fixture
def sample_trace():
    """Create a valid sample Chrome trace for testing."""
    return {
        "traceEvents": [
            {
                "name": "forward",
                "cat": "user_annotation",
                "ph": "X",
                "ts": 1000,
                "dur": 5000,
                "pid": 1,
                "tid": 1,
            },
            {
                "name": "forward",
                "cat": "gpu_user_annotation",
                "ph": "X",
                "ts": 1100,
                "dur": 4500,
                "pid": 1,
                "tid": 2,
            },
        ]
    }


# =============================================================================
# Summary JSON Validation Tests
# =============================================================================

class TestSummaryJsonSchema:
    """Tests for summary.json schema validation."""
    
    def test_valid_summary(self, sample_summary):
        """Test that a valid summary passes validation."""
        errors = validate_summary_schema(sample_summary)
        assert errors == [], f"Unexpected errors: {errors}"
    
    def test_missing_model_field(self, sample_summary):
        """Test detection of missing model field."""
        del sample_summary["model"]
        errors = validate_summary_schema(sample_summary)
        assert "model" in str(errors)
    
    def test_missing_results_field(self, sample_summary):
        """Test detection of missing results field."""
        del sample_summary["results"]
        errors = validate_summary_schema(sample_summary)
        assert "results" in str(errors)
    
    def test_empty_results(self, sample_summary):
        """Test handling of empty results."""
        sample_summary["results"] = {}
        errors = validate_summary_schema(sample_summary)
        assert "empty" in str(errors).lower()
    
    def test_missing_result_fields(self, sample_summary):
        """Test detection of missing fields in result entries."""
        del sample_summary["results"]["Cu_fcc_3x3x3_108atoms"]["qps"]
        errors = validate_summary_schema(sample_summary)
        assert "qps" in str(errors)


class TestTimingValues:
    """Tests for timing value validation."""
    
    def test_positive_timings(self, sample_summary):
        """Test that all timing values are positive."""
        errors = validate_timing_values(sample_summary)
        assert errors == []
    
    def test_negative_mean_ms(self, sample_summary):
        """Test detection of negative mean time."""
        sample_summary["results"]["Cu_fcc_3x3x3_108atoms"]["timeit_mean_ms"] = -1.0
        errors = validate_timing_values(sample_summary)
        assert len(errors) > 0
        assert "negative" in str(errors).lower()
    
    def test_negative_qps(self, sample_summary):
        """Test detection of negative QPS."""
        sample_summary["results"]["Cu_fcc_3x3x3_108atoms"]["qps"] = -10.0
        errors = validate_timing_values(sample_summary)
        assert len(errors) > 0
    
    def test_unreasonable_latency(self, sample_summary):
        """Test detection of unreasonably high latency (>1 hour)."""
        sample_summary["results"]["Cu_fcc_3x3x3_108atoms"]["timeit_mean_ms"] = 4_000_000  # >1 hour
        errors = validate_timing_values(sample_summary)
        assert len(errors) > 0
        assert "unreasonable" in str(errors).lower()
    
    def test_std_greater_than_mean(self, sample_summary):
        """Test warning when std >> mean (unstable measurement)."""
        sample_summary["results"]["Cu_fcc_3x3x3_108atoms"]["timeit_std_ms"] = 100.0
        sample_summary["results"]["Cu_fcc_3x3x3_108atoms"]["timeit_mean_ms"] = 10.0
        warnings = validate_timing_values(sample_summary, return_warnings=True)
        assert any("std" in str(w).lower() for w in warnings)


# =============================================================================
# Trace JSON Validation Tests
# =============================================================================

class TestTraceJsonSchema:
    """Tests for Chrome trace JSON validation."""
    
    def test_valid_trace(self, sample_trace):
        """Test that a valid trace passes validation."""
        errors = validate_trace_schema(sample_trace)
        assert errors == []
    
    def test_missing_trace_events(self):
        """Test detection of missing traceEvents field."""
        invalid_trace = {"other": "data"}
        errors = validate_trace_schema(invalid_trace)
        assert "traceEvents" in str(errors)
    
    def test_empty_trace_events(self, sample_trace):
        """Test handling of empty traceEvents."""
        sample_trace["traceEvents"] = []
        errors = validate_trace_schema(sample_trace)
        assert "empty" in str(errors).lower()
    
    def test_event_missing_required_fields(self, sample_trace):
        """Test detection of events with missing fields."""
        sample_trace["traceEvents"][0] = {"name": "test"}  # Missing ph, dur, etc.
        errors = validate_trace_schema(sample_trace)
        assert len(errors) > 0
    
    def test_invalid_phase(self, sample_trace):
        """Test detection of invalid phase value."""
        sample_trace["traceEvents"][0]["ph"] = "INVALID"
        errors = validate_trace_schema(sample_trace)
        assert "phase" in str(errors).lower() or "ph" in str(errors).lower()


# =============================================================================
# Validation Functions
# =============================================================================

def validate_summary_schema(summary: dict) -> list[str]:
    """
    Validate summary.json schema.
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Check required top-level fields
    for field in SUMMARY_REQUIRED_FIELDS:
        if field not in summary:
            errors.append(f"Missing required field: {field}")
    
    # Check model field
    if "model" in summary:
        model = summary["model"]
        for field in MODEL_REQUIRED_FIELDS:
            if field not in model:
                errors.append(f"Missing model field: {field}")
    
    # Check results
    if "results" in summary:
        results = summary["results"]
        if not results:
            errors.append("Results is empty")
        else:
            for name, data in results.items():
                for field in RESULT_REQUIRED_FIELDS:
                    if field not in data:
                        errors.append(f"Missing field '{field}' in result '{name}'")
    
    return errors


def validate_timing_values(summary: dict, return_warnings: bool = False) -> list[str]:
    """
    Validate timing values are reasonable.
    
    Returns:
        List of errors (and warnings if return_warnings=True)
    """
    errors = []
    warnings = []
    
    MAX_REASONABLE_LATENCY_MS = 3_600_000  # 1 hour
    
    for name, data in summary.get("results", {}).items():
        # Check for negative values
        for field in ["qps", "ns_per_day", "timeit_mean_ms", "timeit_std_ms"]:
            if field in data and data[field] < 0:
                errors.append(f"Negative {field} in '{name}': {data[field]}")
        
        # Check for unreasonable latency
        if "timeit_mean_ms" in data and data["timeit_mean_ms"] > MAX_REASONABLE_LATENCY_MS:
            errors.append(
                f"Unreasonable latency in '{name}': {data['timeit_mean_ms']}ms (>1 hour)"
            )
        
        # Warning: unstable measurement
        mean_ms = data.get("timeit_mean_ms", 0)
        std_ms = data.get("timeit_std_ms", 0)
        if mean_ms > 0 and std_ms > mean_ms:
            warnings.append(
                f"Unstable measurement in '{name}': std ({std_ms}ms) > mean ({mean_ms}ms)"
            )
    
    if return_warnings:
        return errors + warnings
    return errors


def validate_trace_schema(trace: dict) -> list[str]:
    """
    Validate Chrome trace JSON schema.
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Check for traceEvents field
    if "traceEvents" not in trace:
        errors.append("Missing required field: traceEvents")
        return errors
    
    events = trace["traceEvents"]
    if not events:
        errors.append("traceEvents is empty")
        return errors
    
    # Valid phase values in Chrome trace format
    VALID_PHASES = {"B", "E", "X", "i", "I", "C", "b", "n", "e", "s", "t", "f", "P", "N", "O", "D", "M"}
    
    # Check each event
    for idx, event in enumerate(events):
        # Required fields for most events
        if "name" not in event:
            errors.append(f"Event {idx}: missing 'name'")
        
        if "ph" not in event:
            errors.append(f"Event {idx}: missing 'ph' (phase)")
        elif event["ph"] not in VALID_PHASES:
            errors.append(f"Event {idx}: invalid phase '{event['ph']}'")
        
        # For complete events (X), duration is required
        if event.get("ph") == "X" and "dur" not in event:
            errors.append(f"Event {idx}: complete event (ph=X) missing 'dur'")
    
    return errors


# =============================================================================
# Integration Test: Validate Existing Output Files
# =============================================================================

class TestExistingOutputFiles:
    """
    Tests that validate existing profiling output files.
    
    These tests are skipped if no output files exist.
    Use pytest --output-dir=/path/to/outputs to specify a directory.
    """
    
    @pytest.fixture
    def output_dirs(self, project_root):
        """Find existing output directories."""
        # Common output locations
        candidates = [
            project_root / "profile_traces",
            project_root / "results",
            project_root / "test_output_sevenn",
            project_root / "test_output_mace",
            project_root / "test_output_esen",
        ]
        
        existing = []
        for d in candidates:
            if d.exists() and d.is_dir():
                existing.append(d)
        
        return existing
    
    def test_find_and_validate_summaries(self, output_dirs):
        """Find and validate all summary.json files."""
        if not output_dirs:
            pytest.skip("No output directories found")
        
        summary_files = []
        for d in output_dirs:
            summary_files.extend(d.rglob("summary.json"))
        
        if not summary_files:
            pytest.skip("No summary.json files found")
        
        all_errors = []
        for summary_path in summary_files:
            with open(summary_path) as f:
                summary = json.load(f)
            
            schema_errors = validate_summary_schema(summary)
            timing_errors = validate_timing_values(summary)
            
            if schema_errors or timing_errors:
                all_errors.append({
                    "file": str(summary_path),
                    "schema_errors": schema_errors,
                    "timing_errors": timing_errors,
                })
        
        assert not all_errors, f"Validation errors found:\n{json.dumps(all_errors, indent=2)}"
    
    def test_find_and_validate_traces(self, output_dirs):
        """Find and validate all trace.json files."""
        if not output_dirs:
            pytest.skip("No output directories found")
        
        trace_files = []
        for d in output_dirs:
            trace_files.extend(d.rglob("*.trace.json"))
        
        if not trace_files:
            pytest.skip("No trace.json files found")
        
        all_errors = []
        for trace_path in trace_files:
            with open(trace_path) as f:
                trace = json.load(f)
            
            errors = validate_trace_schema(trace)
            if errors:
                all_errors.append({
                    "file": str(trace_path),
                    "errors": errors,
                })
        
        assert not all_errors, f"Validation errors found:\n{json.dumps(all_errors, indent=2)}"
