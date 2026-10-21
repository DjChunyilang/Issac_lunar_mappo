"""Bind complete stage contract tests to the producing run and source state."""

import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from _common import ROOT
from planner_evidence import file_sha256


def require_contract_tests(junit, run, test_files):
    path = Path(junit).resolve()
    if path.parent != (run.path / "metrics").resolve():
        raise ValueError("Contract JUnit must belong to the current run")
    provenance = json.loads(path.with_suffix(".source.json").read_text())
    sources = {name: file_sha256(ROOT / name) for name in test_files}
    if provenance.get("source_snapshot") != run.identity["source_snapshot"] or provenance.get("test_sources") != sources:
        raise ValueError("Contract source provenance mismatch")
    collected = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts=", *test_files],
                               cwd=ROOT, text=True, capture_output=True, check=False)
    if collected.returncode:
        raise ValueError("Contract test collection failed: " + collected.stdout + collected.stderr)
    expected = set()
    for line in collected.stdout.splitlines():
        if any(line.startswith(name + "::") for name in test_files):
            parts = line.strip().split("::")
            classname = parts[0].removesuffix(".py").replace("/", ".")
            if len(parts) > 2:classname += "." + ".".join(parts[1:-1])
            expected.add((classname, parts[-1]))
    cases = list(ET.parse(path).iter("testcase"))
    actual = [(case.get("classname"), case.get("name")) for case in cases]
    errors = [identity for identity, case in zip(actual, cases)
              if any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))]
    if not expected or set(actual) != expected or len(actual) != len(set(actual)) or errors:
        raise ValueError(f"Incomplete or failing contract JUnit: missing={sorted(expected - set(actual))}, errors={errors}")
    return {"passed": True, "tests": len(actual), "failures": 0, "skipped": 0,
            "junit": str(path), "junit_sha256": file_sha256(path), "test_sources": sources,
            "source_matches": True}
