"""Run the complete pipeline: python scripts/run_all.py  (about 20 min on one CPU core; ML step dominates).
Use --skip-ml to reuse existing results/tables/T2x ML outputs."""
import subprocess, sys
from pathlib import Path
here = Path(__file__).parent
steps = ["01_generate_datasets.py", "02_statistics.py", "03_ml_benchmark.py", "04_screening.py",
         "05_make_figures.py", "06_build_excel.py"]
if "--skip-ml" in sys.argv:
    steps.remove("03_ml_benchmark.py")
for s in steps:
    print(f"=== {s}", flush=True)
    subprocess.run([sys.executable, str(here / s)], check=True, cwd=here)
print("Done. Recalculate the workbook formulas with LibreOffice/Excel on first open (values are cached after recalc).")
