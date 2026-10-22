"""tools/run_dashboard.py: stdlib verification, extraction outside the project, local-only serving.
Uses the SYNTHETIC fixture bundle (app/test-fixtures/synthetic_bundle.zip; test data, never results)."""
import hashlib
import importlib.util
import json
import shutil
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "app" / "test-fixtures" / "synthetic_bundle.zip"
if not FIXTURE.exists():                      # e.g. the Colab code bundle, which does not ship app/
    pytest.skip("app/test-fixtures/synthetic_bundle.zip not present", allow_module_level=True)
spec = importlib.util.spec_from_file_location("run_dashboard", ROOT / "tools" / "run_dashboard.py")
RD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RD)


@pytest.fixture
def spaced(tmp_path):
    d = tmp_path / "folder with spaces" / "Downloads (copy)"
    d.mkdir(parents=True)
    z = d / "headpath_dashboard_synthetic copy.zip"
    shutil.copy2(FIXTURE, z)
    return z


def _rewrite(src: Path, dst: Path, change):
    with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w") as zo:
        for n in zi.namelist():
            out = change(n, zi.read(n))
            if out is not None:
                zo.writestr(n, out)


def test_fixture_verifies_with_stdlib_only(spaced):
    v = RD.verify_zip(spaced)
    assert v["status"] == "PASSED" and not v["errors"]
    assert v["files_checked"] > 50 and v["arrays_checked"] > 0 and v["json_strict"] > 10


def test_tampered_member_and_unlisted_file_fail(spaced, tmp_path):
    bad = tmp_path / "bad.zip"
    _rewrite(spaced, bad, lambda n, b: b.replace(b"EXPLORATORY", b"exploratory") if n.endswith("analysis.json") else b)
    v = RD.verify_zip(bad)
    assert v["status"] == "FAILED" and any("analysis.json" in e for e in v["errors"])
    extra = tmp_path / "extra.zip"
    top = zipfile.ZipFile(spaced).namelist()[0].split("/")[0]
    shutil.copy2(spaced, extra)
    with zipfile.ZipFile(extra, "a") as z:
        z.writestr(f"{top}/injected.json", "{}")
    assert any("unlisted" in e for e in RD.verify_zip(extra)["errors"])


def test_corrupted_array_is_detected(spaced, tmp_path):
    """Re-sign the manifest so only the array-content check can catch the change."""
    bad = tmp_path / "bad_array.zip"
    with zipfile.ZipFile(spaced) as zi:
        names = zi.namelist()
        top = names[0].split("/")[0]
        arr = next(n for n in names if n.endswith(".bin.gz"))
        import gzip
        data = bytearray(gzip.decompress(zi.read(arr)))
        data[0] ^= 1
        new = gzip.compress(bytes(data), mtime=0)
        man = json.loads(zi.read(f"{top}/manifest.json"))
        man["files"][arr[len(top) + 1:]] = {"size": len(new), "sha256": hashlib.sha256(new).hexdigest()}
    _rewrite(spaced, bad, lambda n, b: new if n == arr else json.dumps(man).encode() if n.endswith("/manifest.json") else b)
    v = RD.verify_zip(bad)
    assert v["status"] == "FAILED" and any("sha256_raw" in e for e in v["errors"])


def test_extracts_outside_the_project_once_and_never_overwrites(spaced, tmp_path):
    cache = tmp_path / "app data" / "VeriHGN dashboard"
    before = sorted(p.name for p in spaced.parent.iterdir())
    session, data_dir = RD.prepare(spaced, cache)
    assert session["verified"] and data_dir.exists() and (data_dir / "manifest.json").exists()
    assert sorted(p.name for p in spaced.parent.iterdir()) == before          # nothing written next to the ZIP
    assert hashlib.sha256(spaced.read_bytes()).hexdigest() == session["zip_sha256"]
    mtime = (data_dir / "manifest.json").stat().st_mtime_ns
    s2, d2 = RD.prepare(spaced, cache)                                          # reused, not re-extracted
    assert d2 == data_dir and (d2 / "manifest.json").stat().st_mtime_ns == mtime
    with pytest.raises(SystemExit, match="inside the project"):
        RD.prepare(spaced, ROOT / "app" / "cache")
    with pytest.raises(SystemExit, match="original ZIP"):
        RD.prepare(spaced, spaced.parent)


def test_server_is_local_and_serves_only_dist_and_bundle(spaced, tmp_path):
    if not (RD.DIST / "index.html").exists():
        pytest.skip("app/dist not built")
    session, data_dir = RD.prepare(spaced, tmp_path / "cache dir")
    handler = type("H", (RD.Handler,), {"dist": RD.DIST, "data": data_dir, "session": json.dumps(session).encode()})
    httpd = RD.Server(("127.0.0.1", 0), handler)
    assert httpd.server_address[0] == "127.0.0.1"
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert b'<div id="root">' in urllib.request.urlopen(f"{base}/").read()
        assert json.loads(urllib.request.urlopen(f"{base}/data/session.json").read())["zip_sha256"] == session["zip_sha256"]
        assert urllib.request.urlopen(f"{base}/data/manifest.json").read() == (data_dir / "manifest.json").read_bytes()
        arr = next(data_dir.rglob("*.bin.gz"))
        r = urllib.request.urlopen(f"{base}/data/{arr.relative_to(data_dir).as_posix()}")
        assert r.headers.get("Content-Encoding") is None and r.read() == arr.read_bytes()   # served as raw bytes
        for bad in ("/data/../../../Windows/win.ini", "/data/%2e%2e/%2e%2e/secret", "/../tools/run_dashboard.py"):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(base + bad)
            assert e.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_failed_verification_is_served_as_unverified(spaced, tmp_path):
    bad = tmp_path / "bad bundle.zip"
    _rewrite(spaced, bad, lambda n, b: b.replace(b"EXPLORATORY", b"exploratory") if n.endswith("analysis.json") else b)
    session, data_dir = RD.prepare(bad, tmp_path / "cache")
    assert session["verified"] is False and session["verification"]["status"] == "FAILED"
    assert session["verification"]["errors"]


def test_modified_cached_extraction_is_moved_aside_and_extracted_again(spaced, tmp_path):
    cache = tmp_path / "cache dir"
    s1, d1 = RD.prepare(spaced, cache)
    assert s1["verified"]
    rel = "headpath/report.json"
    original = (d1 / rel).read_bytes()
    # tamper with a cached file AND its manifest entry, so the browser's own check against the served manifest would pass
    (d1 / rel).write_bytes(original.replace(b'"VALID"', b'"VALID "', 1))
    man = json.loads((d1 / "manifest.json").read_bytes())
    new = (d1 / rel).read_bytes()
    man["files"][rel] = {"size": len(new), "sha256": hashlib.sha256(new).hexdigest()}
    (d1 / "manifest.json").write_text(json.dumps(man))
    (d1 / "extra.json").write_text("{}")
    assert RD.check_extracted(d1, spaced, d1.name)
    s2, d2 = RD.prepare(spaced, cache)
    assert s2["verified"] and d2 == d1
    assert (d2 / rel).read_bytes() == original and not (d2 / "extra.json").exists()
    assert RD.check_extracted(d2, spaced, d2.name) == []
    aside = [p for p in cache.iterdir() if ".modified_" in p.name]
    assert len(aside) == 1 and (aside[0] / d1.name / "extra.json").exists()     # kept, not deleted
