"""Export the HEADPATH results (+ notebook-07 context) into a validated offline-dashboard bundle (read-only on
the sources). Normally run from notebook 09 on Colab; works on any copy of the Drive folder.

usage:
  python tools/export_headpath_dashboard.py --root /content/drive/MyDrive/VeriHGN --out /content/export/headpath_dashboard_<ts>
  python tools/export_headpath_dashboard.py --verify <bundle.zip>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verihgn_repro.export import dashboard_bundle as DB  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="VeriHGN Drive root (contains reports/ and runs/)")
    ap.add_argument("--out", help="new output directory; the zip is written next to it")
    ap.add_argument("--report", default="08_headpath_experiment.json")
    ap.add_argument("--processed", default="auto", help="processed root for label provenance ('auto' or a path; "
                                                        "'none' skips that check)")
    ap.add_argument("--no-context07", action="store_true")
    ap.add_argument("--context07-report", default=None)
    ap.add_argument("--verify", help="verify an existing bundle zip and exit")
    a = ap.parse_args(argv)
    if a.verify:
        v = DB.verify_bundle(a.verify)
        print(json.dumps(v, indent=1))
        return 0 if v["status"] == "PASSED" else 1
    res = DB.export(a.root, a.out, report_name=a.report,
                    processed_root=None if a.processed == "none" else a.processed,
                    context07=not a.no_context07, context07_report=a.context07_report)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
