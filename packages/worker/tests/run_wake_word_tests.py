#!/usr/bin/env python
"""Run all wake word detection tests."""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all wake word detection related tests."""
    # Get the root directory
    root_dir = Path(__file__).parent.parent.parent.parent
    
    test_files = [
        # Unit tests
        "packages/shared/tests/domain/test_wake_word_model.py",
        "packages/shared/tests/infrastructure/test_wake_word_repository.py",
        "packages/worker/tests/test_enhanced_ffmpeg_processor.py",
        "packages/worker/tests/test_wake_word_detection.py",
        "packages/worker/tests/test_stream_processing_with_wake_words.py",
        
        # Integration tests
        "packages/worker/tests/integration/test_wake_word_detection_integration.py",
    ]
    
    print("Running wake word detection tests...")
    print("=" * 60)
    
    failed_tests = []
    
    for test_file in test_files:
        full_path = root_dir / test_file
        if not full_path.exists():
            print(f"⚠️  Test file not found: {test_file}")
            continue
        
        print(f"\n📋 Running: {test_file}")
        print("-" * 40)
        
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(full_path), "-v"],
            cwd=root_dir,
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            print("✅ PASSED")
        else:
            print("❌ FAILED")
            failed_tests.append(test_file)
            print("\nError output:")
            print(result.stdout)
            print(result.stderr)
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Total test files: {len(test_files)}")
    print(f"Passed: {len(test_files) - len(failed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
        return 1
    else:
        print("\n✅ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(run_tests())