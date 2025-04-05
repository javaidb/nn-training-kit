# Changelog

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
- Added config_loader.py for loading YAML configuration files
- Added sample config.yaml for testing
- Added test for loading configuration from YAML
- Added PyYAML dependency

### Added
- Added accuracy_tolerance parameter to TrainingModule
- Added accuracy_tolerance to TrainerConfig
- Added tests for accuracy_tolerance parameter

### Changed
- Updated FloatHyperparameter documentation to mention accuracy_tolerance use case

### Added
- Initial release of nn-training-kit
- Basic training module with PyTorch Lightning integration
- Support for custom loss functions
- Hyperparameter tuning capabilities
- Logging and metrics tracking

### Changed
- Updated version number for initial release

### Added
- Initial project setup
- Basic project structure
- Core training functionality 