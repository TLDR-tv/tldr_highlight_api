#!/usr/bin/env python3
"""Generate a detailed test coverage report for the TL;DR Highlight API."""

import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def analyze_test_coverage() -> Dict[str, Dict[str, any]]:
    """Analyze test coverage by examining source and test files."""
    
    # Define layers and their paths
    layers = {
        "domain": {
            "source_path": "src/domain",
            "test_path": "tests/unit/domain",
            "modules": defaultdict(lambda: {"has_tests": False, "files": []})
        },
        "application": {
            "source_path": "src/application",
            "test_path": "tests/unit/application",
            "modules": defaultdict(lambda: {"has_tests": False, "files": []})
        },
        "infrastructure": {
            "source_path": "src/infrastructure",
            "test_path": "tests/unit/infrastructure",
            "modules": defaultdict(lambda: {"has_tests": False, "files": []})
        },
        "api": {
            "source_path": "src/api",
            "test_path": "tests/api",
            "modules": defaultdict(lambda: {"has_tests": False, "files": []})
        }
    }
    
    # Analyze each layer
    for layer_name, layer_info in layers.items():
        source_path = project_root / layer_info["source_path"]
        test_path = project_root / layer_info["test_path"]
        
        # Find all Python files in source
        if source_path.exists():
            for py_file in source_path.rglob("*.py"):
                if "__pycache__" not in str(py_file) and not py_file.name.startswith("__"):
                    relative_path = py_file.relative_to(source_path)
                    module_name = str(relative_path.parent / relative_path.stem).replace("/", ".")
                    layer_info["modules"][module_name]["files"].append(str(relative_path))
                    
                    # Check if corresponding test exists
                    test_file_name = f"test_{py_file.name}"
                    potential_test_paths = [
                        test_path / relative_path.parent / test_file_name,
                        test_path / test_file_name,
                    ]
                    
                    for test_file_path in potential_test_paths:
                        if test_file_path.exists():
                            layer_info["modules"][module_name]["has_tests"] = True
                            break
    
    return layers


def generate_report(coverage_data: Dict[str, Dict[str, any]]) -> None:
    """Generate and print the coverage report."""
    
    print("=" * 80)
    print("TEST COVERAGE ANALYSIS REPORT")
    print("=" * 80)
    print()
    
    total_modules = 0
    tested_modules = 0
    
    for layer_name, layer_info in coverage_data.items():
        modules = layer_info["modules"]
        layer_total = len(modules)
        layer_tested = sum(1 for m in modules.values() if m["has_tests"])
        
        total_modules += layer_total
        tested_modules += layer_tested
        
        coverage_pct = (layer_tested / layer_total * 100) if layer_total > 0 else 0
        
        print(f"\n{layer_name.upper()} LAYER")
        print("-" * 40)
        print(f"Total modules: {layer_total}")
        print(f"Tested modules: {layer_tested}")
        print(f"Coverage: {coverage_pct:.1f}%")
        
        # List untested modules
        untested = [name for name, info in modules.items() if not info["has_tests"]]
        if untested:
            print(f"\nUntested modules ({len(untested)}):")
            for module in sorted(untested)[:10]:  # Show first 10
                print(f"  - {module}")
            if len(untested) > 10:
                print(f"  ... and {len(untested) - 10} more")
    
    # Overall summary
    overall_coverage = (tested_modules / total_modules * 100) if total_modules > 0 else 0
    
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total modules: {total_modules}")
    print(f"Tested modules: {tested_modules}")
    print(f"Untested modules: {total_modules - tested_modules}")
    print(f"Overall coverage: {overall_coverage:.1f}%")
    print()
    
    # Risk assessment
    if overall_coverage < 30:
        risk_level = "CRITICAL"
        risk_color = "🔴"
    elif overall_coverage < 60:
        risk_level = "HIGH"
        risk_color = "🟡"
    elif overall_coverage < 80:
        risk_level = "MEDIUM"
        risk_color = "🟡"
    else:
        risk_level = "LOW"
        risk_color = "🟢"
    
    print(f"Risk Level: {risk_color} {risk_level}")
    print()
    
    # Recommendations
    print("RECOMMENDATIONS:")
    print("-" * 40)
    if coverage_data["domain"]["modules"]:
        domain_coverage = sum(1 for m in coverage_data["domain"]["modules"].values() if m["has_tests"]) / len(coverage_data["domain"]["modules"]) * 100
        if domain_coverage < 50:
            print("1. CRITICAL: Domain layer has minimal test coverage. This should be the top priority.")
    
    if coverage_data["application"]["modules"]:
        app_coverage = sum(1 for m in coverage_data["application"]["modules"].values() if m["has_tests"]) / len(coverage_data["application"]["modules"]) * 100
        if app_coverage < 50:
            print("2. CRITICAL: Application use cases are untested. Business logic verification is missing.")
    
    print("3. Create test directories for missing layers (domain, application)")
    print("4. Implement test factories for all domain entities")
    print("5. Add integration tests for critical workflows")
    print()


def check_test_infrastructure() -> None:
    """Check the state of test infrastructure."""
    
    print("\nTEST INFRASTRUCTURE CHECK")
    print("-" * 40)
    
    # Check for test directories
    test_dirs = [
        "tests/unit/domain",
        "tests/unit/application",
        "tests/unit/infrastructure",
        "tests/integration",
        "tests/e2e"
    ]
    
    for test_dir in test_dirs:
        path = project_root / test_dir
        exists = "✅" if path.exists() else "❌"
        print(f"{exists} {test_dir}")
    
    # Check for test utilities
    print("\nTest Utilities:")
    utilities = [
        ("tests/conftest.py", "Pytest configuration"),
        ("tests/factories.py", "Test data factories"),
        ("tests/fixtures.py", "Shared fixtures"),
        ("tests/utils.py", "Test utilities")
    ]
    
    for util_path, description in utilities:
        path = project_root / util_path
        exists = "✅" if path.exists() else "❌"
        print(f"{exists} {util_path} - {description}")
    
    print()


def check_critical_untested_files() -> None:
    """Identify critical files without tests."""
    
    print("\nCRITICAL UNTESTED FILES")
    print("-" * 40)
    
    critical_patterns = [
        ("**/auth*.py", "Authentication/Authorization"),
        ("**/security*.py", "Security"),
        ("**/payment*.py", "Payments/Billing"),
        ("**/webhook*.py", "Webhooks"),
        ("**/stream_processing*.py", "Core Business Logic"),
        ("**/highlight_detection*.py", "Core Business Logic"),
    ]
    
    for pattern, category in critical_patterns:
        print(f"\n{category}:")
        source_files = list(project_root.glob(f"src/{pattern}"))
        
        for source_file in source_files:
            if "__pycache__" not in str(source_file):
                # Check if test exists
                test_name = f"test_{source_file.name}"
                has_test = any(project_root.glob(f"tests/**/test_{source_file.name}"))
                
                status = "✅" if has_test else "❌"
                relative_path = source_file.relative_to(project_root)
                print(f"  {status} {relative_path}")


if __name__ == "__main__":
    print("Analyzing test coverage for TL;DR Highlight API...")
    print()
    
    # Analyze coverage
    coverage_data = analyze_test_coverage()
    
    # Generate report
    generate_report(coverage_data)
    
    # Check infrastructure
    check_test_infrastructure()
    
    # Check critical files
    check_critical_untested_files()
    
    print("\nFor detailed coverage with percentages, run: pytest --cov=src --cov-report=html")