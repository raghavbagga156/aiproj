"""Launch the offline HEADPATH dashboard on a verified export bundle. Python standard library only (no numpy,
no Node.js): works with any Python 3.10+.

  run_dashboard.bat "C:\\path with spaces\\headpath_dashboard_<ts>.zip"
  python tools/run_dashboard.py [ZIP] [--port 8765] [--no-browser] [--cache-dir DIR] [--check-only]

1. Verifies the export ZIP before anything is shown: the zip CRCs; every member listed in manifest.json with the
   listed size and sha256 and nothing unlisted; every .json parses as STRICT JSON (no NaN/Infinity); every array
   decodes (gzip, byte un-shuffle) to its recorded sha256; manifest and integrity status PASSED.
2. Extracts it into a per-user cache keyed by the ZIP's sha256 (default %LOCALAPPDATA%\\VeriHGN-dashboard\\bundles\\),
   never into the project folder or a Drive folder, never over an existing different extraction. A cached
   extraction is re-checked file by file against the ZIP on every launch; a modified one is moved aside (never
   deleted) and the ZIP is extracted again.
3. Serves the prebuilt app (app/dist) and the bundle (under /data/) on 127.0.0.1 only, and opens the browser.
   If verification FAILED the app is still served, but it receives verified=false and withholds every result.
The ZIP and the experiment folders are only read.
"""
import argparse
import gzip
import hashlib
import http.server
import json
import os
import shutil
import socketserver
import sys
import threading
import time
import webbrowser
import zipfile
from pathlib import Path

LAUNCHER_VERSION = "1.1.0"
ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "app" / "dist"
MARKER = ".verihgn_verified.json"


def _reject(c):
    raise ValueError(f"non-strict JSON constant {c}")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unshuffle(planes: bytes, itemsize: int) -> bytes:
    n = len(planes) // itemsize
    out = bytearray(len(planes))
    for b in range(itemsize):
        out[b::itemsize] = planes[b * n:(b + 1) * n]
    return bytes(out)


def _array_metas(z: zipfile.ZipFile, top: str) -> dict:
    metas = {}
    runs = json.loads(z.read(f"{top}/headpath/runs/index.json"))
    for r in runs.values():
        for arrs in r["maps"].values():
            for m in arrs.values():
                metas[m["file"]] = m
    maps = json.loads(z.read(f"{top}/headpath/maps/index.json"))
    for e in maps.values():
        for m in e["arrays"].values():
            metas[m["file"]] = m
    return metas


def verify_zip(zip_path: Path) -> dict:
    """Independent stdlib verification of a bundle zip (mirrors export.dashboard_bundle.verify_bundle)."""
    errors, n_files, n_arr, n_json = [], 0, 0, 0
    try:
        with zipfile.ZipFile(zip_path) as z:
            bad = z.testzip()
            if bad is not None:
                errors.append(f"zip CRC error in {bad}")
            names = [n for n in z.namelist() if not n.endswith("/")]
            tops = {n.split("/")[0] for n in names}
            if len(tops) != 1:
                return {"status": "FAILED", "errors": [f"expected one top-level folder, found {sorted(tops)[:5]}"],
                        "files_checked": 0, "arrays_checked": 0, "json_strict": 0, "top": None}
            top = tops.pop()
            man = json.loads(z.read(f"{top}/manifest.json"), parse_constant=_reject)
            listed = set(man["files"]) | {"manifest.json"}
            present = {n[len(top) + 1:] for n in names}
            if present != listed:
                errors.append(f"unlisted or missing files: {sorted(present ^ listed)[:10]}")
            metas = _array_metas(z, top)
            for rel, info in man["files"].items():
                if rel not in present:
                    continue
                data = z.read(f"{top}/{rel}")
                n_files += 1
                if len(data) != info["size"] or hashlib.sha256(data).hexdigest() != info["sha256"]:
                    errors.append(f"{rel}: size/sha256 differs from manifest.json")
                    continue
                if rel.endswith(".json"):
                    try:
                        json.loads(data, parse_constant=_reject)
                        n_json += 1
                    except ValueError as e:
                        errors.append(f"{rel}: not strict JSON ({e})")
                if rel.endswith(".bin.gz"):
                    m = metas.get(rel)
                    if m is None:
                        errors.append(f"{rel}: array not described in the run/map index")
                        continue
                    try:
                        raw = unshuffle(gzip.decompress(data), m["itemsize"])
                        if len(raw) != m["nbytes_raw"] or hashlib.sha256(raw).hexdigest() != m["sha256_raw"]:
                            errors.append(f"{rel}: decoded array differs from its sha256_raw")
                        else:
                            n_arr += 1
                    except (OSError, EOFError, ValueError) as e:
                        errors.append(f"{rel}: cannot decode ({e})")
            integ = json.loads(z.read(f"{top}/integrity_report.json"), parse_constant=_reject)
            if man.get("status") != "PASSED":
                errors.append(f"manifest status {man.get('status')}")
            if integ.get("status") != "PASSED":
                errors.append(f"integrity report status {integ.get('status')}")
    except (zipfile.BadZipFile, KeyError, ValueError, OSError) as e:
        return {"status": "FAILED", "errors": [f"{type(e).__name__}: {e}"], "files_checked": n_files,
                "arrays_checked": n_arr, "json_strict": n_json, "top": None}
    return {"status": "PASSED" if not errors else "FAILED", "errors": errors, "files_checked": n_files,
            "arrays_checked": n_arr, "json_strict": n_json, "top": top}


def default_cache() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    return (Path(base) if base else Path.home() / ".cache") / "VeriHGN-dashboard" / "bundles"


def _inside(child: Path, parent: Path) -> bool:
    child, parent = child.resolve(), parent.resolve()
    return child == parent or parent in child.parents


def check_cache_location(cache: Path, zip_path: Path) -> None:
    """Never extract into the research project, a Drive folder, or the folder that holds the original ZIP."""
    if _inside(cache, ROOT):
        raise SystemExit(f"refusing: the cache {cache} is inside the project folder {ROOT}")
    low = str(cache.resolve()).lower()
    if "mydrive" in low or "google drive" in low or "googledrive" in low:
        raise SystemExit(f"refusing: the cache {cache} looks like a Google Drive folder")
    if cache.resolve() == zip_path.parent.resolve():
        raise SystemExit(f"refusing: the cache {cache} is the folder that holds the original ZIP")


def check_extracted(bundle_dir: Path, zip_path: Path, top: str) -> list:
    """Differences between an extracted bundle and the ZIP it came from: every member must be present with identical
    bytes (sha256), and nothing else may be present. For a verified ZIP the members are exactly the manifest's files,
    so the served manifest.json, which the browser checks every file against, is itself checked here."""
    problems = []
    members = {}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.startswith(f"{top}/"):
                continue
            h = hashlib.sha256()
            with z.open(info) as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            members[info.filename[len(top) + 1:]] = (info.file_size, h.hexdigest())
    present = {p.relative_to(bundle_dir).as_posix() for p in bundle_dir.rglob("*") if p.is_file()}
    extra = present - set(members)
    if extra:
        problems.append(f"files not in the ZIP: {sorted(extra)[:5]}")
    for rel, (size, sha) in members.items():
        p = bundle_dir / rel
        if not p.is_file():
            problems.append(f"{rel}: missing")
        elif p.stat().st_size != size or sha256_file(p) != sha:
            problems.append(f"{rel}: differs from the ZIP")
    return problems


def _move_aside(target: Path, why: str) -> None:
    aside = target.with_name(f"{target.name}.{why}_{time.strftime('%Y%m%d_%H%M%S')}")
    target.rename(aside)              # never delete, never overwrite
    print(f"previous extraction moved aside: {aside}")


def extract(zip_path: Path, zip_sha: str, cache: Path, top: str) -> Path:
    """Extract once per ZIP sha256 into cache/<sha16>/; reuse an extraction only if it still matches the ZIP file by
    file; never overwrite."""
    target = cache / zip_sha[:16]
    marker = target / MARKER
    if target.exists():
        try:
            rec = json.loads(marker.read_text(encoding="utf-8"))
            if rec.get("zip_sha256") == zip_sha and (target / top / "manifest.json").exists():
                problems = check_extracted(target / top, zip_path, top)
                if not problems:
                    return target / top
                print(f"cached extraction no longer matches the ZIP ({len(problems)} problems, e.g. {problems[0]}); extracting again")
                _move_aside(target, "modified")
        except (OSError, ValueError):
            pass
        if target.exists():
            _move_aside(target, "incomplete")
    tmp = cache / f"{zip_sha[:16]}.tmp_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)            # our own unfinished temp directory only
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            dest = (tmp / info.filename).resolve()
            if not _inside(dest, tmp):
                raise SystemExit(f"refusing unsafe path in zip: {info.filename}")
        z.extractall(tmp)
    (tmp / MARKER).write_text(json.dumps({"zip_sha256": zip_sha, "zip_name": zip_path.name,
                                          "extracted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}),
                              encoding="utf-8")
    problems = check_extracted(tmp / top, zip_path, top)
    if problems:
        raise SystemExit(f"extraction does not match the ZIP: {problems[:3]}")
    tmp.rename(target)
    return target / top


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "VeriHGN-dashboard"
    dist: Path = DIST
    data: Path | None = None
    session: bytes = b"{}"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, body: bytes, ctype: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        from urllib.parse import unquote
        path = unquote(path)
        if path == "/data/session.json":
            return self._send(self.session, "application/json")
        if path.startswith("/data/"):
            if self.data is None:
                return self._send(b"no bundle", "text/plain", 404)
            return self._file(self.data, path[len("/data/"):])
        rel = path.lstrip("/") or "index.html"
        return self._file(self.dist, rel)

    def _file(self, root: Path, rel: str):
        p = (root / rel).resolve()
        if not _inside(p, root) or not p.is_file():
            return self._send(b"not found", "text/plain", 404)
        types = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css",
                 ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon",
                 ".csv": "text/csv", ".md": "text/markdown", ".yaml": "text/plain"}
        # .bin.gz arrays are served as plain bytes (no Content-Encoding): the app decodes them itself
        return self._send(p.read_bytes(), types.get(p.suffix, "application/octet-stream"))


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = False


def find_default_zip() -> Path | None:
    d = Path.home() / "Downloads"
    zips = sorted((p for p in d.glob("headpath_dashboard_*.zip")), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def prepare(zip_path: Path, cache: Path) -> tuple[dict, Path | None]:
    zip_path = zip_path.resolve()
    if not zip_path.is_file():
        raise SystemExit(f"export ZIP not found: {zip_path}")
    check_cache_location(cache, zip_path)
    print(f"export: {zip_path}")
    zsha = sha256_file(zip_path)
    print(f"sha256: {zsha}")
    print("verifying (manifest, sha256 of every file, strict JSON, lossless array decode) ...", flush=True)
    t0 = time.time()
    v = verify_zip(zip_path)
    print(f"verification {v['status']} in {time.time() - t0:.1f}s: {v['files_checked']} files, "
          f"{v['arrays_checked']} arrays, {v['json_strict']} JSON files")
    for e in v["errors"][:20]:
        print("  -", e)
    data_dir = None
    if v["top"]:
        data_dir = extract(zip_path, zsha, cache, v["top"])
        print(f"bundle: {data_dir}")
    session = {"app": "verihgn-headpath-dashboard", "zip_name": zip_path.name, "zip_path": str(zip_path),
               "zip_sha256": zsha, "zip_bytes": zip_path.stat().st_size, "verified": v["status"] == "PASSED",
               "verification": {k: v[k] for k in ("status", "errors", "files_checked", "arrays_checked", "json_strict")},
               "extract_dir": str(data_dir) if data_dir else "", "launched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "launcher_version": LAUNCHER_VERSION}
    return session, data_dir


def serve(session: dict, data_dir: Path | None, port: int, open_browser: bool, dist: Path = DIST):
    if not (dist / "index.html").exists():
        raise SystemExit(f"the built app is missing: {dist / 'index.html'}")
    handler = type("H", (Handler,), {"dist": dist, "data": data_dir,
                                     "session": json.dumps(session).encode("utf-8")})
    last = None
    for p in range(port, port + 20):
        try:
            httpd = Server(("127.0.0.1", p), handler)
            break
        except OSError as e:
            last = e
    else:
        raise SystemExit(f"no free port from {port}: {last}")
    url = f"http://127.0.0.1:{httpd.server_address[1]}/#/overview"
    print(f"\nDashboard: {url}\n(local only; press Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", nargs="?", help="headpath_dashboard_<ts>.zip (default: the newest one in Downloads)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--cache-dir", default=None, help="where verified bundles are extracted (default: per-user app data)")
    ap.add_argument("--check-only", action="store_true", help="verify and extract, print the result, do not serve")
    a = ap.parse_args(argv)
    zp = Path(a.zip) if a.zip else find_default_zip()
    if zp is None:
        raise SystemExit("no export ZIP given and none found in Downloads (headpath_dashboard_*.zip)")
    session, data_dir = prepare(zp, Path(a.cache_dir) if a.cache_dir else default_cache())
    if a.check_only:
        print(json.dumps(session, indent=1))
        return 0 if session["verified"] else 1
    if not session["verified"]:
        print("\nVERIFICATION FAILED: the dashboard opens in its integrity-failure state and shows no results.")
    serve(session, data_dir, a.port, not a.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
