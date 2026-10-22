"""Helpers used by the Colab notebooks: Drive layout, dependency install,
dataset access checks, raw-data fetching and extraction.

Authentication is only ever the user's own `drive.mount` popup; nothing here
stores or requests credentials.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .data import circuitnet as cn
from .utils import hardware


@dataclass(frozen=True)
class DriveLayout:
    root: Path = Path("/content/drive/MyDrive/VeriHGN")
    local_raw: Path = Path("/content/circuitnet_ispd15")     # fast local disk, rebuilt each session
    local_archives: Path = Path("/content/archives")

    @property
    def processed(self): return self.root / "processed" / "ispd15"
    @property
    def splits(self): return self.root / "splits"
    @property
    def graph_cache(self):
        # Graphs are ~30 GB per experiment variant for the 383 samples: kept on local disk only, never on Drive.
        return Path("/content/graph_cache")
    @property
    def runs(self): return self.root / "runs"
    @property
    def raw_cache(self): return self.root / "raw_cache" / "ispd15"
    @property
    def reports(self): return self.root / "reports"
    @property
    def export(self): return self.root / "export"

    def processed_for(self, prep_hash: str) -> Path:
        """Processed data lives under its preprocessing hash, so a changed assumption never
        overwrites earlier data on Drive."""
        return self.processed / prep_hash

    def ensure(self):
        """Create missing folders only; never deletes or overwrites anything."""
        for p in (self.processed, self.splits, self.runs, self.reports, self.export):
            p.mkdir(parents=True, exist_ok=True)
        return self


def install_dependencies(verbose: bool = True) -> dict:
    """Install PyG (+ torch_scatter when a matching wheel exists) for Colab's torch."""
    import torch
    report = {"torch": torch.__version__, "cuda": torch.version.cuda}
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_geometric", "gdown", "pyyaml"], check=True)
    base = torch.__version__.split("+")[0]
    tag = ("cu" + torch.version.cuda.replace(".", "")) if torch.version.cuda else "cpu"
    major_minor = ".".join(base.split(".")[:2])
    url = f"https://data.pyg.org/whl/torch-{major_minor}.0+{tag}.html"
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary=:all:", "torch_scatter",
                        "-f", url], capture_output=True, text=True)
    report["torch_scatter_wheel"] = "installed" if r.returncode == 0 else f"none for {url} (shim will be used)"
    from .official import scatter_impl
    report["scatter_impl"] = scatter_impl()
    if verbose:
        print(json.dumps(report, indent=1))
    return report


def access_check(drive_root="/content/drive") -> dict:
    """Locate CircuitNet-ISPD15 through the mounted Drive and verify archive sizes."""
    found = cn.find_ispd15_on_drive(drive_root)
    rep = {"found": found, "archives": []}
    if found["path"]:
        for a in cn.ARCHIVES.values():
            rep["archives"].append(cn.verify_archive(cn.local_archive_path(found["path"], a), a))
    rep["ok"] = bool(found["path"]) and all(x["ok"] for x in rep["archives"] if x["archive"] != "def")
    if not found["path"]:
        rep["instructions"] = (
            "Open https://drive.google.com/drive/folders/" + cn.ROOT_FOLDER_ID +
            " -> folder menu -> Organize -> Add shortcut -> My Drive, then re-run this cell. "
            "Alternatively use fetch_archives(method='gdown').")
    return rep


def storage_report(layout: DriveLayout) -> dict:
    """Free space on Colab local disk and on the mounted Drive (Drive figure comes
    from the FUSE mount's statfs and reflects the account quota as reported by
    Drive; treat it as approximate)."""
    return hardware.disk_report(["/content", "/content/drive"])


NEEDED_FOR_SCOPE = ("graph_information", "placement_gcell", "placement_micron", "routability")


def fetch_archives(layout: DriveLayout, method: str = "drive", ispd15_dir=None, cache_on_drive: bool = False,
                   keys=NEEDED_FOR_SCOPE) -> dict:
    """Make the needed archives readable locally. method='drive' reads the shortcut
    (optionally copying to the Drive raw_cache), method='gdown' downloads to /content."""
    out = {}
    layout.local_archives.mkdir(parents=True, exist_ok=True)
    for k in keys:
        a = cn.ARCHIVES[k]
        t0 = time.time()
        cached = layout.raw_cache / a.filename
        if cached.exists() and cached.stat().st_size == a.size_bytes:
            out[k] = str(cached)
        elif method == "drive":
            src = cn.local_archive_path(ispd15_dir, a)
            if cache_on_drive:
                cached.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, cached)
                out[k] = str(cached)
            else:
                out[k] = str(src)      # stream directly from the shortcut
        elif method == "gdown":
            p = cn.download_with_gdown(a, layout.local_archives)
            if cache_on_drive:
                cached.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, cached)
            out[k] = str(p)
        else:
            raise ValueError(method)
        print(f"{k}: {out[k]} ({time.time() - t0:.0f}s)")
    return out


def extract_for_scope(archive_paths: dict, layout: DriveLayout, scope: str = "official9") -> dict:
    """Extract only what the scope needs into layout.local_raw."""
    designs = set(cn.SCOPES[scope])
    root = layout.local_raw
    counts = {}
    t0 = time.time()
    counts["graph_information"] = cn.extract_members(archive_paths["graph_information"], root,
                                                     prefixes=["graph_infromation"], log_every=0)
    # sample names of the scope come from the gcell archive listing itself
    counts["placement_gcell"] = cn.extract_members(archive_paths["placement_gcell"], root,
                                                   prefixes=["instance_placement_gcell"], log_every=0)
    samples = [s for s in cn.RawLayout(root).samples() if cn.parse_sample_name(s)["design"] in designs]
    # gcell extraction includes all scopes (sample names are only known after reading); drop out-of-scope ones
    for s in cn.RawLayout(root).samples():
        if cn.parse_sample_name(s)["design"] not in designs:
            (root / "instance_placement_gcell" / s).unlink()
    counts["placement_micron"] = cn.extract_members(archive_paths["placement_micron"], root,
                                                    prefixes=["instance_placement_micron"],
                                                    sample_names=samples, log_every=0)
    counts["routability"] = cn.extract_members(archive_paths["routability"], root,
                                               prefixes=list(cn.ROUTABILITY_MEMBERS.values()),
                                               sample_names=samples, log_every=1000)
    counts["samples_in_scope"] = len(samples)
    counts["seconds"] = round(time.time() - t0)
    return counts


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


class DrivePersistenceError(RuntimeError):
    """Flushing the Drive mount, or verifying the flushed files, failed."""


def _sha256_file(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def flush_drive(paths=(), remount_verify: bool = True, mountpoint: str = "/content/drive", drive_module=None) -> dict:
    """Last step of a notebook that wrote to Drive (as notebook 05's last cell). Writes under /content/drive are
    uploaded asynchronously; google.colab.drive.flush_and_unmount() blocks until everything written in this
    session has been uploaded, then unmounts.
      1. records the sha256 of each of `paths` (the content that must persist), BEFORE unmounting;
      2. flush_and_unmount();
      3. remount_verify: drive.mount() again and re-hash each file through the new mount.
    Nothing may be written to /content/drive after step 2 (it would land on the VM disk, not on Drive), so the
    caller must finish every report write first. Any failure raises DrivePersistenceError; never silent."""
    if drive_module is None:
        from google.colab import drive as drive_module
    paths = [Path(p) for p in paths]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise DrivePersistenceError(f"files to persist are missing before the flush: {missing}")
    hashes = {str(p): _sha256_file(p) for p in paths}
    try:
        drive_module.flush_and_unmount()
    except Exception as e:  # noqa: BLE001 - re-raised with the cause
        raise DrivePersistenceError(f"drive.flush_and_unmount() failed: {type(e).__name__}: {e}") from e
    rec = {"flushed_and_unmounted": True, "sha256_before_flush": hashes, "remount_verified": None}
    if remount_verify:
        try:
            drive_module.mount(mountpoint)
        except Exception as e:  # noqa: BLE001
            raise DrivePersistenceError(f"remount for verification failed: {type(e).__name__}: {e}") from e
        bad = {p: h for p, h in hashes.items() if not Path(p).exists() or _sha256_file(p) != h}
        if bad:
            raise DrivePersistenceError(f"after flush + remount these files are missing or differ: {sorted(bad)}")
        rec["remount_verified"] = True
    return rec
