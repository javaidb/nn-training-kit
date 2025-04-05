# Testing Infrastructure

This directory contains all testing-related files for the nn-training-kit library. These files are not included in the distributed package and are only used for development and testing purposes.

## Directory Structure

- `configs/`: Sample configuration files for testing
- `data/`: Sample data for testing
- `test_*.py`: Test files for different components of the library

## Running Tests

To run the tests:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run tests with coverage report
pytest --cov=nn_training_kit --cov-report=term-missing
```

## Sample Data

The `data/` directory contains synthetic data for testing purposes. This data is not meant for production use but is designed to demonstrate the functionality of the library.

## Sample Configurations

The `configs/` directory contains YAML configuration files that demonstrate how to configure the library for different use cases. These can be used as templates for your own configurations. 