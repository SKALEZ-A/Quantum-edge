"""
Test Runner for Quantum Edge AI Platform

Comprehensive test execution and reporting system.
"""

import unittest
import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import coverage

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests import run_all_tests, run_unit_tests, run_integration_tests, run_performance_tests

class TestRunner:
    """Advanced test runner with reporting and analysis"""

    def __init__(self, test_directory: str = "tests"):
        self.test_directory = Path(test_directory)
        self.results = {}
        self.start_time = None
        self.end_time = None

    def discover_tests(self) -> list:
        """Discover all test files"""
        test_files = []

        # Find all test files
        for test_file in self.test_directory.glob("test_*.py"):
            if test_file.name != "test_runner.py":  # Exclude this file
                test_files.append(test_file)

        return sorted(test_files)

    def run_test_suite(self, test_type: str = "all", verbose: bool = False,
                      coverage_report: bool = False) -> dict:
        """Run test suite with specified configuration"""
        self.start_time = time.time()

        if test_type == "unit":
            result = run_unit_tests()
        elif test_type == "integration":
            result = run_integration_tests()
        elif test_type == "performance":
            result = run_performance_tests()
        else:
            result = run_all_tests(verbose=verbose, coverage=coverage_report)

        self.end_time = time.time()

        # Store results
        self.results = {
            'test_type': test_type,
            'result': result,
            'execution_time': self.end_time - self.start_time,
            'timestamp': datetime.utcnow().isoformat(),
            'test_files': len(self.discover_tests())
        }

        return self.results

    def run_individual_test_file(self, test_file: str) -> dict:
        """Run a specific test file"""
        if not test_file.endswith('.py'):
            test_file += '.py'

        test_path = self.test_directory / test_file
        if not test_path.exists():
            return {'error': f'Test file {test_file} not found'}

        self.start_time = time.time()

        # Run the specific test file
        result = subprocess.run([
            sys.executable, '-m', 'pytest',
            str(test_path), '-v', '--tb=short'
        ], capture_output=True, text=True)

        self.end_time = time.time()

        return {
            'test_file': test_file,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'execution_time': self.end_time - self.start_time,
            'success': result.returncode == 0
        }

    def generate_report(self, output_format: str = "console") -> str:
        """Generate test report"""
        if not self.results:
            return "No test results available"

        report_data = {
            'summary': {
                'total_tests': self.results.get('test_files', 0),
                'execution_time': self.results.get('execution_time', 0),
                'timestamp': self.results.get('timestamp'),
                'test_type': self.results.get('test_type', 'unknown')
            },
            'results': self.results.get('result', {}),
            'metrics': self._calculate_metrics()
        }

        if output_format == "json":
            return json.dumps(report_data, indent=2)
        elif output_format == "console":
            return self._format_console_report(report_data)
        else:
            return str(report_data)

    def _calculate_metrics(self) -> dict:
        """Calculate test metrics"""
        result = self.results.get('result', {})

        metrics = {
            'success_rate': 0.0,
            'average_execution_time': self.results.get('execution_time', 0),
            'tests_per_second': 0.0
        }

        if result.get('success'):
            metrics['success_rate'] = 100.0

            execution_time = self.results.get('execution_time', 0)
            if execution_time > 0:
                # Estimate tests per second (rough approximation)
                estimated_tests = 100  # Assume ~100 tests in full suite
                metrics['tests_per_second'] = estimated_tests / execution_time

        return metrics

    def _format_console_report(self, report_data: dict) -> str:
        """Format report for console output"""
        lines = []
        lines.append("=" * 60)
        lines.append("QUANTUM EDGE AI PLATFORM - TEST REPORT")
        lines.append("=" * 60)

        summary = report_data['summary']
        lines.append(f"Test Type: {summary['test_type'].upper()}")
        lines.append(f"Total Test Files: {summary['total_tests']}")
        lines.append(".2f")
        lines.append(f"Timestamp: {summary['timestamp']}")

        lines.append("\n" + "-" * 40)
        lines.append("RESULTS")
        lines.append("-" * 40)

        results = report_data['results']
        if results.get('success'):
            lines.append("✅ OVERALL STATUS: PASSED")
        else:
            lines.append("❌ OVERALL STATUS: FAILED")

        if 'stdout' in results:
            lines.append(f"\nSTDOUT:\n{results['stdout']}")

        if 'stderr' in results:
            lines.append(f"\nSTDERR:\n{results['stderr']}")

        lines.append("\n" + "-" * 40)
        lines.append("METRICS")
        lines.append("-" * 40)

        metrics = report_data['metrics']
        lines.append(".1f")
        lines.append(".2f")
        lines.append(".1f")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)

    def save_report(self, filename: str, format: str = "json"):
        """Save test report to file"""
        report = self.generate_report(format)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"Report saved to {filename}")

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description='Quantum Edge AI Platform Test Runner')
    parser.add_argument('--type', choices=['all', 'unit', 'integration', 'performance'],
                       default='all', help='Type of tests to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--coverage', action='store_true', help='Enable coverage reporting')
    parser.add_argument('--file', help='Run specific test file')
    parser.add_argument('--output', help='Output report file')
    parser.add_argument('--format', choices=['console', 'json'], default='console',
                       help='Report output format')

    args = parser.parse_args()

    # Initialize test runner
    runner = TestRunner()

    print("🧪 Quantum Edge AI Platform - Test Suite")
    print("=" * 50)

    if args.file:
        # Run specific test file
        print(f"Running specific test file: {args.file}")
        result = runner.run_individual_test_file(args.file)

        if result.get('error'):
            print(f"❌ Error: {result['error']}")
            sys.exit(1)

        print(f"Test completed in {result['execution_time']:.2f} seconds")
        if result['success']:
            print("✅ Test PASSED")
        else:
            print("❌ Test FAILED")
            print(f"STDOUT:\n{result['stdout']}")
            print(f"STDERR:\n{result['stderr']}")

    else:
        # Run test suite
        print(f"Running {args.type} test suite...")

        result = runner.run_test_suite(
            test_type=args.type,
            verbose=args.verbose,
            coverage_report=args.coverage
        )

        # Generate and display report
        report = runner.generate_report(args.format)
        print(report)

        # Save report if requested
        if args.output:
            runner.save_report(args.output, args.format)

        # Exit with appropriate code
        if result.get('result', {}).get('success', False):
            print("\n🎉 All tests passed!")
            sys.exit(0)
        else:
            print("\n💥 Some tests failed!")
            sys.exit(1)

if __name__ == '__main__':
    main()
