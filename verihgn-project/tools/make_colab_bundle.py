"""Package the project code for Colab: writes dist/verihgn_code.zip.

Upload the zip to Google Drive at MyDrive/VeriHGN/code/verihgn_code.zip; every
notebook unpacks it. Contains code, configs, tests and the official
third_party/VeriHGN sources (without its .git directory). No data.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "configs", "tests", "third_party/VeriHGN", "notebooks", "tools", "docs", "pyproject.toml",
           "requirements.txt", "README.md"]
EXCLUDE_PARTS = {".git", "__pycache__", ".pytest_cache", "fixtures"}


def main():
    out = ROOT / "dist" / "verihgn_code.zip"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            p = ROOT / item
            files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
            for f in files:
                rel = f.relative_to(ROOT)
                if EXCLUDE_PARTS & set(rel.parts) or f.suffix in {".pt", ".npz", ".pyc"}:
                    continue
                z.write(f, rel.as_posix())
                n += 1
        # .git is excluded, so record the official commit for official.official_commit()
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from verihgn_repro.official import official_commit
        z.writestr("third_party/VeriHGN/OFFICIAL_COMMIT", official_commit() + "\n")
    print(f"wrote {out} ({n} files, {out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
