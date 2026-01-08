# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.1] - 2026-01-02

### Added
- Makefile for common tasks
- CHANGELOG.md for tracking changes
- ADDITIONAL_SUGGESTIONS.md with more improvement ideas
- Enhanced start_all_bots.sh with automatic config validation

### Changed
- Improved error messages in validate_config.py
- Enhanced backup_state.sh to backup watchlists

## [2.0.0] - 2026-01-02

### Added
- Setup script (`setup.sh`) for easy onboarding
- Configuration validation script (`validate_config.py`)
- Backup script (`backup_state.sh`) for state files
- Health check endpoint (`health_check.py`)
- Version tracking (`VERSION` file)
- `.editorconfig` for consistent code formatting
- Comprehensive documentation:
  - `FILE_ANALYSIS.md` - File necessity analysis
  - `CLEANUP_SUMMARY.md` - Cleanup report
  - `IMPROVEMENT_SUGGESTIONS.md` - Improvement roadmap
  - `QUICK_WINS.md` - Quick improvements guide
  - `QUICK_WINS_IMPLEMENTED.md` - Implementation summary

### Changed
- Replaced print statements with proper logging in core modules
- Updated README.md to reference `run_bots.py` instead of `main.py`
- Enhanced `start_all_bots.sh` with config validation
- Added version display to `run_bots.py`

### Fixed
- Fixed documentation inconsistencies
- Removed duplicate documentation files (13 files)
- Cleaned up unnecessary files (~36 files/folders)
- Updated `.gitignore` with comprehensive patterns

### Security
- Added sensitive data filtering in logs (already existed, documented)
- Improved `.gitignore` patterns for runtime artifacts

## [1.0.0] - 2025-XX-XX

### Added
- Initial release
- Core bot framework
- Multiple trading strategies (11 bots)
- Risk management system
- Emergency stop protection
- Portfolio-level risk controls
