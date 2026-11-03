import runpy
from pathlib import Path


def test_example_runs() -> None:
    example_path: Path = Path(__file__).parent.parent / "examples" / "quadratic.py"
    runpy.run_path(str(example_path), run_name="__main__")
