# HEI Presentations

This repository hosts Quarto-based slide presentations for HEI (Higher Education Institution) courses, with a focus on data computation topics. Currently, it includes materials on Spark IO and file layouts for efficient data processing.

## Features

- Interactive Quarto presentations (.qmd files) with embedded code execution.
- Python 3.13+ support for data examples using libraries like PySpark.
- Custom styling via SCSS for consistent branding.
- Modular structure for easy addition of new topics.

## Prerequisites

- [Quarto](https://quarto.org/docs/get-started/) (CLI tool for rendering).
- Python 3.13+ (managed via `.python-version` and tools like pyenv).
- [uv](https://docs.astral.sh/uv/) for fast Python dependency management.

## Installation

1. Clone the repository:

   ```
   git clone <repo-url>
   cd hei-presentations
   ```

3. Install dependencies:

   ```
   uv sync
   ```

   This installs packages from `pyproject.toml` and locks them in `uv.lock`.

4. Verify setup:
   - Run `uv run quarto --version` to check Quarto.
   - Run `uv run python --version` to confirm Python 3.13+.

## Usage

### Rendering Presentations

To generate static HTML/PDF slides:

```
uv run quarto render presentations/302.1-data-computation/spark-io-presentation.qmd
```

Output appears in the same directory as where the .qmd file is

### Live Preview

For development with auto-reload:

```
uv run quarto preview presentations/302.1-data-computation/spark-io-presentation.qmd
```

This serves the presentation at `http://localhost:xxxx` (port shown on start).

### Running Python Code in Presentations

Quarto executes Python cells during rendering. Ensure dependencies (e.g., PySpark) are added by running `uv add pyspark` before rendering.

Example: Add to a .qmd file:

```python
import pyspark
# Your Spark code here
```

## Project Structure

```
hei-presentations/
├── content/                    # Supporting Markdown files
│   └── 302.1-data-computation/
│       └── spark-io-file-layout.md
├── presentations/              # Quarto presentation sources
│   └── 302.1-data-computation/
│       └── spark-io-presentation.qmd
├── styles/                     # Custom CSS/SCSS
│   └── 302.1-data-computation/
│       └── spark-io-presentation-style.scss
├── pyproject.toml              # Python project config
├── uv.lock                     # Dependency lockfile
├── .python-version             # Python version pin
└── README.md                   # This file
```

## Adding New Presentations

1. Create a new directory under `presentations/` and `content/` (e.g., `303-data-analysis/`).
2. Add a `.qmd` file in `presentations/<topic>/` with YAML frontmatter:

   ```yaml
   ---
   title: "New Topic"
   format: revealjs
   ---
   ```

3. Include content, code cells, and reference supporting MD files.
4. Add custom styles in `styles/<topic>/` if needed.
5. Update `pyproject.toml` for any new Python dependencies, then `uv sync`.
6. Render and preview as above.

## Development and Contribution

- Follow code style in [AGENTS.md](AGENTS.md) for Python additions.
- For linting/formatting: Install `ruff` and `black` via `uv add ruff black`, then run `ruff check .` and `black .`.
- No tests yet; consider adding pytest for Python scripts.
- Contributions: Fork, branch, PR with descriptive messages. Focus on educational content enhancements.

## License

This project is licensed under the MIT License (or specify if different). See LICENSE file for details.

## Support

- Issues: [GitHub Issues](https://github.com/sst/opencode/issues) (for opencode tool feedback).
- Quarto Docs: [quarto.org](https://quarto.org/docs/presentations/revealjs/).
- Python Help: Standard PEP 8 and type hints.
