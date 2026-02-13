# Contributing to EgoDex Robotics

Thank you for your interest in contributing! This project is research-focused, but we aim for high-quality, readable code.

## Getting Started

1. **Fork and Clone** the repository.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Data**: Ensure you have access to the Apple EgoDex dataset and place samples in `video_learning_samples/`.

## Code Style

We follow [PEP 8](https://peps.python.org/pep-0008/) and use `ruff` for linting and formatting.

- **Linting**:
  ```bash
  ruff check .
  ```
- **Formatting**:
  ```bash
  ruff format .
  ```

## Pull Requests

1. Create a new branch for your feature or fix.
2. Add detailed docstrings to any new functions.
3. Verify that existing scripts (`3d.py`, `bounding.py`, `generate_depth.py`) still run correctly.
4. Submit a PR with a clear description of your changes.

## Reporting Issues

Please open an issue on GitHub if you encounter bugs or have feature requests. Include details about your environment and the video sample you are using.
