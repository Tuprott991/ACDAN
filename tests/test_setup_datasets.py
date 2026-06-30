import importlib.util
import sys


def _setup_module():
    spec = importlib.util.spec_from_file_location("setup_datasets", "scripts/setup_datasets.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_agentic_benchmark_dry_run_manifest(tmp_path):
    setup = _setup_module()

    manifest = setup.prepare_agentic_benchmarks(
        tmp_path,
        overwrite=False,
        token=None,
        dry_run=True,
    )

    expected = {
        "browsecomp",
        "webvoyager",
        "swe_bench_verified",
        "terminal_bench",
        "mathhay",
        "tau2_bench_data",
        "tau2_bench_hud",
        "mcp_bench",
    }
    assert set(manifest) == expected
    assert manifest["browsecomp"]["original_size"] == 1266
    assert manifest["swe_bench_verified"]["source"] == "SWE-bench/SWE-bench_Verified"
    assert manifest["tau2_bench_data"]["source"] == "HuggingFaceH4/tau2-bench-data"
    assert all(entry["acdan_runnable"] is False for entry in manifest.values())
    assert all(entry["dry_run"] is True for entry in manifest.values())
    assert not (tmp_path / "raw").exists()


def test_agentic_benchmark_selection_validation(tmp_path):
    setup = _setup_module()

    manifest = setup.prepare_agentic_benchmarks(
        tmp_path,
        overwrite=False,
        token=None,
        dry_run=True,
        selected={"webvoyager", "mcp_bench"},
    )

    assert set(manifest) == {"webvoyager", "mcp_bench"}
