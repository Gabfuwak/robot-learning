# RoboCasa Project

_Authors: Gabin Maury, Minh Duc Nguyen, Chenxi Yang_

## Setup

Requires [uv](https://github.com/astral-sh/uv) and Python 3.11.

```bash
git clone --recurse-submodules robot-learning
chmod +x setup.sh && ./setup.sh
```

This will:
1. Initialize git submodules (robosuite, robocasa)
2. Create a virtual environment at `venv/`
3. Install dependencies
4. Download ~10GB of kitchen assets

## Usage

Activate the environment:
```bash
source venv/bin/activate
```

Run the test script:
```bash
python test.py
```
