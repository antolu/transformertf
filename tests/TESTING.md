# Testing Strategy for TransformerTF

## Overview

This document outlines the comprehensive testing strategy for TransformerTF, including unit tests, integration tests, property-based tests, and various pytest techniques.

## Test Structure Analysis

### Current Separation: Unit vs Integration Tests

**✅ Good Separation:**
- **Unit tests** (`tests/`): Test individual components in isolation
- **Integration tests** (`tests/integration/`): Test component interactions and data flow
- Clear directory structure mirroring source code
- Appropriate fixture scoping

**Areas for Improvement:**
- Some integration tests are more like unit tests (testing single components)
- Need more end-to-end integration tests
- Missing cross-model integration tests

## Enhanced Testing Framework

### 1. Pytest Configuration

Enhanced `pyproject.toml` configuration includes:
- **Coverage reporting**: 85% minimum coverage with HTML/XML reports
- **Parallel execution**: Using pytest-xdist for faster test runs
- **Strict markers**: Enforces marker usage for better test organization
- **Test markers**: Categorizes tests by type (unit, integration, property, etc.)

### 2. Test Markers

```python
@pytest.mark.unit          # Isolated component tests
@pytest.mark.integration   # Component interaction tests
@pytest.mark.slow          # Tests taking >1s
@pytest.mark.gpu           # GPU-required tests
@pytest.mark.property      # Property-based tests
@pytest.mark.benchmark     # Performance benchmarks
@pytest.mark.smoke         # Quick validation tests
@pytest.mark.regression    # Regression tests
@pytest.mark.edge_case     # Edge case tests
```

### 3. New Testing Dependencies

```toml
test = [
    "pytest",           # Core testing framework
    "pytest-cov",       # Coverage reporting
    "pytest-xdist",     # Parallel execution
    "pytest-benchmark", # Performance benchmarking
    "pytest-mock",      # Mocking utilities
    "pytest-timeout",   # Test timeout handling
    "hypothesis",       # Property-based testing
    "factoryboy",       # Test data factories
]
```

## Testing Techniques

### 1. Factory Pattern (`tests/factories.py`)

Create test objects using factory_boy:

```python
# Create test data
df = TimeSeriesDataFactory.create(n_samples=500, n_features=3)

# Create model configurations
config = TFTModelFactory.create()

# Create sample batches
batch = create_sample_batch(batch_size=4, ctxt_seq_len=100)
```

### 2. Property-Based Testing (`tests/strategies.py`)

Use Hypothesis for property-based testing:

```python
@given(config=tft_config_strategy())
def test_model_invariants(config):
    model = TemporalFusionTransformerModel(**config)
    # Test invariant properties
```

### 3. Parametrized Tests

Test multiple scenarios efficiently:

```python
@pytest.mark.parametrize(
    "batch_size,seq_len,num_features",
    [(1, 10, 2), (4, 50, 3), (8, 100, 5)]
)
def test_attention_shapes(batch_size, seq_len, num_features):
    # Test with various configurations
```

### 4. Assertion Utilities (`tests/utils.py`)

Enhanced assertion functions:

```python
assert_tensor_shape(tensor, expected_shape)
assert_tensor_finite(tensor)
assert_tensor_not_nan(tensor)
assert_tensor_range(tensor, min_val, max_val)
```

### 5. Mock Testing

Test with mocked dependencies:

```python
@patch('transformertf.models.temporal_fusion_transformer._model.InterpretableMultiHeadAttention')
def test_model_with_mocked_components(mock_attention):
    # Test with mocked internal components
```

### 6. Benchmark Testing

Performance regression testing:

```python
@pytest.mark.benchmark
def test_forward_pass_performance(benchmark):
    result = benchmark(run_forward)
    assert_tensor_finite(result["output"])
```

## Test Organization

### Unit Tests
- **Location**: `tests/`
- **Purpose**: Test individual components in isolation
- **Examples**: Individual layer tests, transform tests, utility function tests

### Integration Tests
- **Location**: `tests/integration/`
- **Purpose**: Test component interactions and data flow
- **Examples**: Model + DataModule tests, full training pipeline tests

### Property-Based Tests
- **Location**: Throughout test suite with `@pytest.mark.property`
- **Purpose**: Test invariant properties across many generated inputs
- **Examples**: Model output shape invariants, mathematical property tests

### Edge Case Tests
- **Location**: Throughout test suite with `@pytest.mark.edge_case`
- **Purpose**: Test boundary conditions and error handling
- **Examples**: Empty sequences, very large batches, invalid inputs

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run specific test types
pytest -m unit
pytest -m integration
pytest -m property

# Run with coverage
pytest --cov=transformertf
```

### Performance Testing
```bash
# Run benchmarks
pytest -m benchmark

# Run slow tests
pytest -m slow
```

### Continuous Integration
```bash
# Run full test suite with coverage
pytest --cov=transformertf --cov-fail-under=85
```

## Best Practices

### 1. Test Isolation
- Use fixtures for setup/teardown
- Avoid test interdependencies
- Use temporary directories for file operations

### 2. Test Data
- Use factories for consistent test data generation
- Create realistic synthetic data for physics applications
- Use property-based testing for edge cases

### 3. Assertions
- Use descriptive assertion messages
- Check both positive and negative cases
- Validate tensor properties (shape, dtype, finite values)

### 4. Performance
- Mark slow tests appropriately
- Use benchmarks for performance regression detection
- Optimize test execution with parallel runs

### 5. Error Handling
- Test exception paths
- Validate error messages
- Check graceful degradation

## Test Coverage Goals

- **Unit Tests**: 90%+ coverage of individual components
- **Integration Tests**: Cover all major workflows
- **Property Tests**: Cover mathematical invariants
- **Edge Cases**: Cover boundary conditions and error paths
- **Performance**: Benchmark critical paths

## Future Enhancements

1. **GPU Testing**: Add GPU-specific test markers and fixtures
2. **Distributed Testing**: Add multi-GPU and distributed training tests
3. **Model Comparison**: Add tests comparing different model architectures
4. **Data Pipeline**: Add comprehensive data loading and preprocessing tests
5. **Hyperparameter**: Add hyperparameter sensitivity tests
6. **Visualization**: Add tests for plotting and visualization functions

## Maintenance

- Regular review of test performance and coverage
- Update property-based test strategies as code evolves
- Maintain factory definitions as models change
- Review and update test markers and categorization
- Keep test documentation current
