# Testing Guide

This document explains how to run the test files for this project.

## Prerequisites

- Python 3.7 or higher installed
- All project dependencies installed (see `requirements.txt`)
- (Recommended) Use a virtual environment

## Running All Tests

From the project root directory, run:

```
python -m unittest discover -s tests
```

This will automatically discover and run all test files in the `tests/` directory.

## Running a Specific Test File

To run a specific test file, use:

```
python -m unittest tests/test_scrapper.py
python -m unittest tests/test_fact_checker.py
```

## Notes

- The tests use mocking, so no real network requests are made.
- Make sure your `.env` file (for API keys) is present if required by the code, but tests are designed to work even if the key is missing (they mock the environment).
- If you add new test files, name them with the `test_*.py` pattern and place them in the `tests/` directory.

## Troubleshooting

- If you see `ModuleNotFoundError`, ensure you are running the command from the project root and your virtual environment is activated.
- If dependencies are missing, install them with:
  ```
  pip install -r requirements.txt
  ```
- If you change environment variables, restart your shell or IDE to ensure changes are picked up. 