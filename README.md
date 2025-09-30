# HEI Presentations

This repository hosts Quarto-based slide presentations for HEI (Higher Education Institution) courses, focusing on machine learning (301-1-ML) and data computation (302.1) topics. It includes materials on SVM, decision trees/random forests, Spark IO, file layouts, and Parquet for efficient data processing.

## Features

- Interactive Quarto presentations (.qmd files) with embedded code execution.
- Python 3.13+ support for data and ML examples using libraries like PySpark and scikit-learn.
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

2. Install dependencies:

   ```
   uv sync
   ```

   This installs packages from `pyproject.toml` and locks them in `uv.lock`.

3. Verify setup:
   - Run `uv run quarto --version` to check Quarto.
   - Run `uv run python --version` to confirm Python 3.13+.

## Usage

### Rendering Presentations

To generate static HTML/PDF slides:

For Spark IO:
```
uv run quarto render presentations/302.1-data-computation/spark-io-presentation.qmd
```

For SVM:
```
uv run quarto render presentations/301-1-ml/svm-presentation.qmd
```

For Decision Trees/Random Forests:
```
uv run quarto render presentations/301-1-ml/decision_trees_random_forests-presentation.qmd
```

Output appears in the same directory as the .qmd file.

To convert HTML to PDF, open the generated HTML, switch to PDF mode, and print from the browser.

### Live Preview

For development with auto-reload (e.g., for Spark IO):
```
uv run quarto preview presentations/302.1-data-computation/spark-io-presentation.qmd
```

This serves at `http://localhost:xxxx` (port shown on start). Adjust path for other presentations.

### Running Python Code in Presentations

Quarto executes Python cells during rendering. Ensure dependencies (e.g., PySpark, scikit-learn) are added via `uv add pyspark scikit-learn` before rendering.

Example in .qmd:
```python
import pyspark
# Spark code here
```

Or for ML:
```python
from sklearn.svm import SVC
# ML code here
```

## Project Structure

```
hei-presentations/
├── content/                    # Supporting Markdown files and scripts
│   ├── 301-1-ml/
│   │   └── svm.md
│   └── 302.1-data-computation/
│       ├── decision_trees_random_forests.md
│       ├── parquet_demo.py
│       └── spark-io-file-layout.md
├── presentations/              # Quarto presentation sources and outputs
│   ├── 301-1-ml/
│   │   ├── decision_trees_random_forests-presentation.qmd
│   │   └── svm-presentation.qmd
│   └── 302.1-data-computation/
│       ├── spark-io-presentation.qmd
│       └── Spark I_O and File Layout_ Partitioning Best Practices & Parquet Introduction.pdf
├── styles/                     # Custom CSS/SCSS
│   └── 302.1-data-computation/
│       └── spark-io-presentation-style.scss
├── pyproject.toml              # Python project config
├── uv.lock                     # Dependency lockfile
├── .gitignore
├── .python-version             # Python version pin
└── README.md                   # This file
```

## Adding New Presentations

1. Create directories under `presentations/`, `content/`, and `styles/` if needed (e.g., `303-data-analysis/`).
2. Add `.qmd` file in `presentations/<topic>/` with YAML frontmatter:
   ```yaml
   ---
   title: "New Topic"
   format: revealjs
   ---
   ```
3. Include content, code cells, and reference MD files.
4. Add custom styles in `styles/<topic>/` if needed.
5. Update `pyproject.toml` for new dependencies, then `uv sync`.
6. Render and preview as above.

## Development and Contribution

- Follow code style in [AGENTS.md](AGENTS.md) for Python additions.
- For linting/formatting: `uv add ruff black`, then `ruff check .` and `black .`.
- No tests yet; consider adding pytest for scripts.
- Contributions: Fork, branch, PR with descriptive messages. Focus on educational enhancements.

## License

MIT License. See LICENSE file for details (add if needed).

## Support

- Issues: [GitHub Issues](https://github.com/sst/opencode/issues) (for opencode feedback).
- Quarto Docs: [quarto.org](https://quarto.org/docs/presentations/revealjs/).
- Python Help: PEP 8 and type hints.
