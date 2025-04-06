# Changelog

## [1.0.0] - 2024-04-05

### Added
- Reorganized testing infrastructure
- Added sample data generation for testing
- Added data loader for testing
- Added sample configuration files
- Added MANIFEST.in to include tests in source distribution

### Changed
- Excluded tests directory from installed package
- Updated test files to use the new configuration structure

### Added
- Initial stable release
- Comprehensive training module with PyTorch Lightning integration
- YAML-based configuration system
- Hyperparameter tuning capabilities
- Accuracy tolerance parameter
- Logging and metrics tracking
- Sample data generation for testing

## [0.1.3] - 2024-03-16

### Added
- Added config_loader.py for loading YAML configuration files
- Added sample config.yaml for testing
- Added test for loading configuration from YAML
- Added PyYAML dependency

## [0.1.2] - 2024-03-16

### Added
- Added accuracy_tolerance parameter to TrainingModule
- Added accuracy_tolerance to TrainerConfig
- Added tests for accuracy_tolerance parameter

### Changed
- Updated FloatHyperparameter documentation to mention accuracy_tolerance use case

## [0.1.1] - 2024-03-16

### Added
- Initial release of nn-training-kit
- Basic training module with PyTorch Lightning integration
- Support for custom loss functions
- Hyperparameter tuning capabilities
- Logging and metrics tracking

### Changed
- Updated version number for initial release

## [0.1.0] - 2024-03-16

### Added
- Initial project setup
- Basic project structure
- Core training functionality 