# Contributing Guide

Welcome to the Quantum Edge AI Platform! We appreciate your interest in contributing to this open-source project. This guide will help you understand how to contribute effectively.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style and Standards](#code-style-and-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Community](#community)

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:

- **Be respectful**: Treat all contributors with respect and kindness
- **Be inclusive**: Welcome people from all backgrounds and experiences
- **Be collaborative**: Work together to solve problems and improve the project
- **Be responsible**: Take ownership of your contributions and their impact

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.8 or higher
- Git
- Basic understanding of quantum computing concepts
- Familiarity with machine learning and edge computing

### Setting Up Development Environment

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub
   # Then clone your fork
   git clone https://github.com/your-username/quantum-edge-ai-platform.git
   cd quantum-edge-ai-platform
   ```

2. **Set Up Development Environment**
   ```bash
   # Follow the development setup guide
   # See docs/tutorials/development_setup.md for detailed instructions
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Run Tests**
   ```bash
   # Ensure everything works
   pytest
   ```

## Development Workflow

### 1. Choose an Issue

- Check the [GitHub Issues](https://github.com/your-org/quantum-edge-ai-platform/issues) page
- Look for issues labeled `good first issue` or `help wanted`
- Comment on the issue to indicate you're working on it

### 2. Create a Feature Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-number-description
```

### 3. Make Changes

- Write clear, focused commits
- Test your changes thoroughly
- Follow the coding standards
- Update documentation as needed

### 4. Run Quality Checks

```bash
# Format code
black .
isort .

# Lint code
flake8 .

# Type check
mypy .

# Run tests
pytest --cov=quantum_edge_ai

# Build documentation
cd docs && make html
```

### 5. Submit Pull Request

- Push your branch to your fork
- Create a pull request from your branch to the main repository
- Fill out the pull request template completely
- Wait for review and address feedback

## Code Style and Standards

### Python Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some additional guidelines:

```python
# Good: Clear, descriptive names
def calculate_quantum_kernel(x_train, x_test, gamma=0.1):
    """Calculate quantum kernel matrix between training and test data."""
    pass

# Bad: Unclear abbreviations
def calc_qkern(xt, xte, g=0.1):
    pass
```

### Import Organization

Use absolute imports and organize them properly:

```python
# Standard library imports
import os
import sys
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
from qiskit import QuantumCircuit

# Local imports
from quantum_edge_ai.quantum_algorithms import QuantumSVM
from .utils import validate_input
```

### Type Hints

Use type hints for all function parameters and return values:

```python
from typing import Dict, List, Optional, Union

def train_quantum_model(
    X: np.ndarray,
    y: np.ndarray,
    n_qubits: int = 4,
    learning_rate: float = 0.01
) -> Dict[str, Union[float, np.ndarray]]:
    """Train a quantum machine learning model."""
    pass
```

### Documentation Strings

Write comprehensive docstrings following Google style:

```python
def quantum_feature_map(
    x: np.ndarray,
    n_qubits: int,
    depth: int = 1
) -> QuantumCircuit:
    """Create a quantum feature map for encoding classical data.

    This function implements various quantum feature mapping techniques
    to encode classical data into quantum states for quantum machine learning.

    Args:
        x: Input classical data array of shape (n_samples, n_features)
        n_qubits: Number of qubits in the quantum circuit
        depth: Depth of the feature map circuit

    Returns:
        QuantumCircuit: The parameterized quantum feature map circuit

    Raises:
        ValueError: If input dimensions are incompatible with n_qubits

    Example:
        >>> import numpy as np
        >>> from qiskit import QuantumCircuit
        >>> x = np.array([[0.1, 0.2], [0.3, 0.4]])
        >>> qfm = quantum_feature_map(x, n_qubits=2)
        >>> isinstance(qfm, QuantumCircuit)
        True
    """
    pass
```

### Error Handling

Handle errors gracefully and provide meaningful error messages:

```python
class QuantumEdgeError(Exception):
    """Base exception for Quantum Edge AI Platform."""
    pass

class QuantumCircuitError(QuantumEdgeError):
    """Exception raised for quantum circuit related errors."""
    pass

def validate_quantum_circuit(circuit: QuantumCircuit) -> None:
    """Validate quantum circuit parameters."""
    if circuit.num_qubits < 1:
        raise QuantumCircuitError("Circuit must have at least 1 qubit")

    if circuit.depth() > 1000:
        raise QuantumCircuitError("Circuit depth exceeds maximum allowed (1000)")

    # Additional validations...
```

## Testing

### Test Structure

Tests are organized in the `tests/` directory:

```
tests/
├── __init__.py
├── test_quantum_algorithms.py
├── test_edge_runtime.py
├── test_privacy_security.py
├── test_api_services.py
├── test_integration.py
├── test_performance.py
└── conftest.py
```

### Writing Tests

Use pytest with descriptive test names:

```python
import pytest
import numpy as np
from quantum_edge_ai.quantum_algorithms.quantum_ml import QuantumSVM

class TestQuantumSVM:
    """Test cases for Quantum Support Vector Machine."""

    def test_initialization(self):
        """Test QuantumSVM initialization with valid parameters."""
        qsvm = QuantumSVM(n_qubits=4, kernel='rbf')

        assert qsvm.n_qubits == 4
        assert qsvm.kernel == 'rbf'

    def test_fit_with_valid_data(self):
        """Test fitting with valid training data."""
        qsvm = QuantumSVM(n_qubits=2)

        # Generate simple training data
        X = np.array([[0, 0], [1, 1]])
        y = np.array([1, -1])

        # Should not raise any exceptions
        qsvm.fit(X, y)

        assert qsvm.is_fitted

    def test_predict_requires_fit(self):
        """Test that predict requires the model to be fitted first."""
        qsvm = QuantumSVM(n_qubits=2)
        X_test = np.array([[0.5, 0.5]])

        with pytest.raises(ModelNotFittedError):
            qsvm.predict(X_test)

    @pytest.mark.parametrize("n_qubits", [1, 2, 4, 8])
    def test_different_qubit_counts(self, n_qubits):
        """Test QuantumSVM with different numbers of qubits."""
        qsvm = QuantumSVM(n_qubits=n_qubits)

        assert qsvm.n_qubits == n_qubits
        # Test that circuit can be created
        circuit = qsvm._create_circuit()
        assert circuit.num_qubits == n_qubits
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_quantum_algorithms.py

# Run specific test class/method
pytest tests/test_quantum_algorithms.py::TestQuantumSVM::test_initialization

# Run tests with coverage
pytest --cov=quantum_edge_ai --cov-report=html

# Run tests in verbose mode
pytest -v

# Run tests in parallel
pytest -n auto

# Run only tests marked with specific marker
pytest -m "slow"  # Requires conftest.py marker configuration
```

### Test Coverage

Maintain high test coverage (>90%):

```bash
# Generate coverage report
pytest --cov=quantum_edge_ai --cov-report=html --cov-report=term

# Check coverage threshold
pytest --cov=quantum_edge_ai --cov-fail-under=90
```

## Documentation

### Documentation Standards

- Use Markdown for all documentation
- Keep documentation up to date with code changes
- Include code examples in docstrings and documentation
- Provide tutorials and guides for complex features

### API Documentation

Document all public APIs:

```python
def run_inference(
    model_id: str,
    input_data: Union[np.ndarray, List[float]],
    options: Optional[Dict[str, Any]] = None
) -> InferenceResult:
    """Run inference using a trained model.

    This function provides a high-level interface for running inference
    on input data using pre-trained models in the platform.

    Args:
        model_id: Unique identifier of the trained model
        input_data: Input data for inference. Can be a numpy array
                   or list of floats.
        options: Optional dictionary of inference options including:
                - precision: Desired numerical precision ('FP32', 'FP16', 'INT8')
                - timeout: Maximum inference time in seconds
                - batch_size: Number of inputs to process in parallel

    Returns:
        InferenceResult object containing:
        - prediction: Model prediction
        - confidence: Prediction confidence score (0-1)
        - processing_time: Time taken for inference in seconds
        - model_info: Information about the model used

    Raises:
        ModelNotFoundError: If model_id does not exist
        InferenceError: If inference fails
        TimeoutError: If inference exceeds timeout

    Example:
        >>> result = run_inference(
        ...     model_id="qsvm_001",
        ...     input_data=[0.1, 0.2, 0.3, 0.4],
        ...     options={"precision": "FP16"}
        ... )
        >>> print(f"Prediction: {result.prediction}")
        >>> print(f"Confidence: {result.confidence:.2f}")
    """
```

### Documentation Updates

When making code changes:

1. **Update docstrings** for modified functions/classes
2. **Update guides** if behavior changes
3. **Add examples** for new features
4. **Update API docs** for public interface changes

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**
   ```bash
   pytest --cov=quantum_edge_ai --cov-fail-under=90
   ```

2. **Format and lint code**
   ```bash
   black .
   isort .
   flake8 .
   mypy .
   ```

3. **Update documentation**
   ```bash
   # Build docs and check for errors
   cd docs && make html
   ```

4. **Write clear commit messages**
   ```bash
   # Good commit message
   git commit -m "feat: add quantum kernel methods to QuantumSVM

   - Implement RBF and polynomial kernels
   - Add kernel parameter validation
   - Update tests for new kernel functionality
   - Add documentation examples"

   # Bad commit message
   git commit -m "update stuff"
   ```

### Pull Request Template

Fill out the pull request template completely:

```markdown
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement

## Related Issues
Fixes #123, Addresses #456

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed
- [ ] All tests pass

## Documentation
- [ ] Docstrings updated
- [ ] User documentation updated
- [ ] API documentation updated
- [ ] Examples added/updated

## Checklist
- [ ] Code follows style guidelines
- [ ] Commit messages are clear and descriptive
- [ ] No sensitive information committed
- [ ] Dependencies updated (if applicable)
- [ ] Migration scripts included (if applicable)
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests, linting, and security checks
2. **Peer Review**: At least one maintainer reviews the code
3. **Approval**: PR is approved and merged by a maintainer
4. **Deployment**: Changes are automatically deployed to staging/production

## Reporting Issues

### Bug Reports

Use the bug report template:

```markdown
**Describe the Bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python Version: [e.g., 3.9.7]
- Platform Version: [e.g., 1.0.0]
- Browser: [e.g., Chrome 91]

**Additional Context**
Add any other context about the problem here.
```

### Feature Requests

Use the feature request template:

```markdown
**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
```

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Slack**: Real-time chat for contributors
- **Mailing List**: Announcements and newsletters

### Getting Help

- **Documentation**: Check the docs first
- **GitHub Issues**: Search existing issues
- **Community Forum**: Ask questions on GitHub Discussions
- **Slack**: Get real-time help from other contributors

### Recognition

Contributors are recognized in several ways:

- **Contributors File**: Added to `CONTRIBUTORS.md`
- **GitHub Recognition**: Listed as a contributor on the repository
- **Release Notes**: Mentioned in release notes for significant contributions
- **Community Events**: Invited to community events and meetups

## Areas for Contribution

### High Priority

- **Quantum Algorithm Improvements**: Enhance existing quantum ML algorithms
- **Edge Optimization**: Improve performance on edge devices
- **Privacy Enhancements**: Strengthen privacy-preserving techniques
- **Documentation**: Improve and expand documentation
- **Testing**: Add comprehensive test coverage

### Good First Issues

- **Documentation Updates**: Fix typos, improve clarity
- **Test Improvements**: Add missing test cases
- **Code Cleanup**: Remove deprecated code, improve style
- **Performance Monitoring**: Add metrics and monitoring
- **Error Handling**: Improve error messages and handling

### Advanced Contributions

- **New Quantum Algorithms**: Implement novel quantum ML techniques
- **Federated Learning**: Enhance distributed learning capabilities
- **Hardware Acceleration**: Optimize for specific hardware platforms
- **Security Research**: Research and implement advanced security features
- **Scalability**: Improve platform scalability and performance

## Code Review Guidelines

### For Reviewers

- **Be constructive**: Focus on improving the code, not criticizing the contributor
- **Be specific**: Point out exact issues and suggest specific solutions
- **Be timely**: Review PRs promptly to keep development moving
- **Be thorough**: Check for edge cases, security issues, and performance problems

### For Contributors

- **Respond to feedback**: Address all review comments thoughtfully
- **Ask questions**: If something is unclear, ask for clarification
- **Be patient**: Reviews take time; follow up politely if needed
- **Learn from feedback**: Use reviews as learning opportunities

## License and Copyright

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (Apache License 2.0). You also confirm that you have the right to license your contributions.

---

Thank you for contributing to the Quantum Edge AI Platform! Your contributions help advance the field of quantum computing and edge AI. 🚀
