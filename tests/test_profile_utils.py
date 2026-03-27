"""
Tests for profile_utils.py
"""
import pytest
import json
import tempfile
from pathlib import Path


class TestExtractOperationTimes:
    """Tests for extract_operation_times_from_trace()"""
    
    def test_empty_trace(self, tmp_path):
        """Test handling of empty trace file."""
        from profile_utils import extract_operation_times_from_trace
        
        trace_file = tmp_path / "empty.json"
        trace_file.write_text('{"traceEvents": []}')
        
        result = extract_operation_times_from_trace(
            trace_path=trace_file,
            active_steps=5,
            tracked_operations=["test_op"],
        )
        assert result == {}
    
    def test_simple_trace(self, tmp_path):
        """Test extraction from simple trace."""
        from profile_utils import extract_operation_times_from_trace
        
        trace_data = {
            "traceEvents": [
                {
                    "name": "test_op",
                    "cat": "user_annotation",
                    "ph": "X",
                    "dur": 1000,  # 1ms in microseconds
                },
                {
                    "name": "test_op",
                    "cat": "gpu_user_annotation",
                    "ph": "X",
                    "dur": 500,
                },
            ]
        }
        
        trace_file = tmp_path / "test.json"
        trace_file.write_text(json.dumps(trace_data))
        
        result = extract_operation_times_from_trace(
            trace_path=trace_file,
            active_steps=1,
            tracked_operations=["test_op"],
        )
        
        assert "test_op" in result
        assert result["test_op"]["cpu_time_ms"] == 1.0  # 1000us = 1ms
        assert result["test_op"]["gpu_time_ms"] == 0.5  # 500us = 0.5ms
    
    def test_file_not_found(self):
        """Test FileNotFoundError for missing trace file."""
        from profile_utils import extract_operation_times_from_trace
        
        with pytest.raises(FileNotFoundError):
            extract_operation_times_from_trace(
                trace_path=Path("/nonexistent/path/trace.json"),
                active_steps=5,
                tracked_operations=["test_op"],
            )
    
    def test_invalid_json(self, tmp_path):
        """Test ValueError for invalid JSON."""
        from profile_utils import extract_operation_times_from_trace
        
        trace_file = tmp_path / "invalid.json"
        trace_file.write_text("not valid json {{{")
        
        with pytest.raises(ValueError):
            extract_operation_times_from_trace(
                trace_path=trace_file,
                active_steps=5,
                tracked_operations=["test_op"],
            )
    
    def test_untracked_operations_ignored(self, tmp_path):
        """Test that operations not in tracked_operations are ignored."""
        from profile_utils import extract_operation_times_from_trace
        
        trace_data = {
            "traceEvents": [
                {"name": "tracked_op", "cat": "user_annotation", "ph": "X", "dur": 1000},
                {"name": "untracked_op", "cat": "user_annotation", "ph": "X", "dur": 2000},
            ]
        }
        
        trace_file = tmp_path / "test.json"
        trace_file.write_text(json.dumps(trace_data))
        
        result = extract_operation_times_from_trace(
            trace_path=trace_file,
            active_steps=1,
            tracked_operations=["tracked_op"],
        )
        
        assert "tracked_op" in result
        assert "untracked_op" not in result
    
    def test_multiple_active_steps(self, tmp_path):
        """Test averaging over multiple active steps."""
        from profile_utils import extract_operation_times_from_trace
        
        # Simulate 3 active steps
        trace_data = {
            "traceEvents": [
                {"name": "op", "cat": "gpu_user_annotation", "ph": "X", "dur": 1000},
                {"name": "op", "cat": "gpu_user_annotation", "ph": "X", "dur": 2000},
                {"name": "op", "cat": "gpu_user_annotation", "ph": "X", "dur": 3000},
            ]
        }
        
        trace_file = tmp_path / "test.json"
        trace_file.write_text(json.dumps(trace_data))
        
        result = extract_operation_times_from_trace(
            trace_path=trace_file,
            active_steps=3,
            tracked_operations=["op"],
        )
        
        # Total: 6000us = 6ms, averaged over 3 steps = 2ms
        assert result["op"]["gpu_time_ms"] == 2.0
        assert result["op"]["count"] == 1  # 3 events / 3 steps = 1 per step


class TestSynchronize:
    """Tests for synchronize()"""
    
    def test_cpu_device(self):
        """Test synchronize on CPU (should be no-op)."""
        from profile_utils import synchronize
        
        # Should not raise
        synchronize("cpu")
    
    @pytest.mark.skipif(
        not __import__("torch").cuda.is_available(),
        reason="CUDA not available"
    )
    def test_cuda_device(self):
        """Test synchronize on CUDA."""
        from profile_utils import synchronize
        
        # Should not raise
        synchronize("cuda")


class TestGetQPS:
    """Tests for get_qps()"""
    
    def test_simple_function(self):
        """Test QPS measurement with simple function."""
        from profile_utils import get_qps
        import time
        
        def dummy_fn():
            time.sleep(0.001)  # 1ms
        
        qps, ns_per_day, mean_ms, std_ms = get_qps(
            dummy_fn,
            device="cpu",
            warmups=2,
            timeiters=3,
            repeats=2,
        )
        
        assert mean_ms > 0
        assert qps > 0
        assert std_ms >= 0
        assert ns_per_day > 0
