"""CircuitNet-ISPD15 on Google Drive: identifiers, discovery, download and
selective extraction.

Every ID and byte size below was observed during the Phase 0 audit
(docs/DATASET_AUDIT.md). Nothing here holds credentials: access is either the
user's own `drive.mount` session or the public link share.
"""
import os
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

ROOT_FOLDER_ID = "1GjW-1LBx1563bg3pHQGvhcEyK2A9sYUB"      # "CircuitNet"
ISPD15_FOLDER_ID = "1-icpZOIBDjaLvcnzy310TGoZXome5Zd-"    # "CircuitNet-ISPD15"


@dataclass(frozen=True)
class Archive:
    key: str
    filename: str
    file_id: str
    size_bytes: int
    drive_relpath: str   # path relative to the CircuitNet root folder


ARCHIVES = {
    a.key: a for a in [
        Archive("graph_information", "graph_infromation.tar.gz", "1WiBhEEGXBzmL5ueQataKbazHzRQIneCd",
                123_753_781, "CircuitNet-ISPD15/graph_features/graph_infromation.tar.gz"),
        Archive("placement_gcell", "instance_placement_gcell.tar.gz", "1EYcXapMF-c8WOcAFwATrMxwzWI-xitXQ",
                1_955_989_756, "CircuitNet-ISPD15/graph_features/instance_placement_gcell.tar.gz"),
        Archive("placement_micron", "instance_placement_micron.tar.gz", "1dZrouqyFua22_r_VAaN9JBvtkgsmUJdu",
                2_724_127_925, "CircuitNet-ISPD15/graph_features/instance_placement_micron.tar.gz"),
        Archive("routability", "routability_features.tar.gz", "1skK2NloJe3kPiHaTL7aCLBtWGAjo4ZXU",
                7_588_018_472, "CircuitNet-ISPD15/routability_features.tar.gz"),
        Archive("def", "DEF.tar.gz", "1ZzCqGrrrGwRDl6BT0PHdNUGf6a5yLG6T",
                11_397_704_623, "CircuitNet-ISPD15/DEF.tar.gz"),
    ]
}

# Routability sub-folders this project reads (6 of 17; see DATASET_AUDIT §3.5).
ROUTABILITY_MEMBERS = {
    "gr_h_overflow": "routability_features/congestion/congestion_global_routing/overflow_based/congestion_GR_horizontal_overflow",
    "gr_v_overflow": "routability_features/congestion/congestion_global_routing/overflow_based/congestion_GR_vertical_overflow",
    "cell_density": "routability_features/cell_density",
    "rudy": "routability_features/RUDY/RUDY",
    "rudy_pin": "routability_features/RUDY/RUDY_pin",
    "macro_map": "routability_features/macro_map",
}

# Designs: CircuitNet name -> number of placement samples (observed).
SAMPLES_PER_DESIGN = {
    "mgc_des_perf_a": 45, "mgc_edit_dist_a": 45, "mgc_fft_a": 45, "mgc_fft_b": 43,
    "mgc_matrix_mult_a": 45, "mgc_matrix_mult_b": 44, "mgc_matrix_mult_c": 44,
    "mgc_pci_bridge32_a": 27, "mgc_pci_bridge32_b": 45,
    "mgc_superblue11_a": 30, "mgc_superblue12": 30, "mgc_superblue14": 30,
    "mgc_superblue16_a": 30, "mgc_superblue19": 30,
}
GRAPH_ONLY_DESIGNS = ("mgc_des_perf_b", "mgc_fft_2")   # graph info but no samples

SCOPES = {
    # official code: sorted design folders without "superblue" (data/dataset.py:32-33)
    "official9": sorted(d for d in SAMPLES_PER_DESIGN if "superblue" not in d),
    "all14": sorted(SAMPLES_PER_DESIGN),
}

_SAMPLE_RE = re.compile(r"^(?P<idx>\d+)-(?P<design>.+)-util_(?P<util>[\d.]+?)(?:\.npy)?$")


def parse_sample_name(name: str) -> dict:
    m = _SAMPLE_RE.match(os.path.basename(name))
    if not m:
        raise ValueError(f"not a CircuitNet-ISPD15 sample name: {name}")
    return {"sample": f"{m['idx']}-{m['design']}-util_{m['util']}", "index": int(m["idx"]),
            "design": m["design"], "util": float(m["util"])}


# --------------------------------------------------------------------- discovery
def find_ispd15_on_drive(drive_root="/content/drive") -> dict:
    """Locate CircuitNet-ISPD15 through the user's mounted Drive.

    Tries, in order: a `.shortcut-targets-by-id/<ROOT_FOLDER_ID>` entry (how
    Drive-FUSE exposes shortcut targets), common shortcut names in MyDrive, and a
    bounded scan of MyDrive for a folder holding the expected archive names.
    Returns {"path": ..., "method": ...} or {"path": None, "tried": [...]}.
    """
    root = Path(drive_root)
    tried = []
    candidates = [
        (root / ".shortcut-targets-by-id" / ROOT_FOLDER_ID / "CircuitNet" / "CircuitNet-ISPD15", "shortcut-targets-by-id"),
        (root / ".shortcut-targets-by-id" / ROOT_FOLDER_ID / "CircuitNet-ISPD15", "shortcut-targets-by-id"),
        (root / "MyDrive" / "CircuitNet" / "CircuitNet-ISPD15", "MyDrive/CircuitNet shortcut"),
        (root / "MyDrive" / "CircuitNet-ISPD15", "MyDrive/CircuitNet-ISPD15"),
    ]
    for path, method in candidates:
        tried.append(str(path))
        if _looks_like_ispd15(path):
            return {"path": str(path), "method": method}
    mydrive = root / "MyDrive"
    if mydrive.exists():
        for dirpath, dirnames, _ in os.walk(mydrive):
            depth = Path(dirpath).relative_to(mydrive).parts
            if len(depth) >= 3:
                dirnames[:] = []
                continue
            if Path(dirpath).name == "CircuitNet-ISPD15" and _looks_like_ispd15(Path(dirpath)):
                return {"path": dirpath, "method": "scan of MyDrive"}
        tried.append(f"scan of {mydrive} (depth 3)")
    return {"path": None, "tried": tried}


def _looks_like_ispd15(path: Path) -> bool:
    return (path / "routability_features.tar.gz").exists() and (path / "graph_features").is_dir()


def verify_archive(path, archive: Archive) -> dict:
    size = Path(path).stat().st_size if Path(path).exists() else None
    return {"archive": archive.key, "path": str(path), "size": size,
            "expected": archive.size_bytes, "ok": size == archive.size_bytes}


def local_archive_path(ispd15_dir, archive: Archive) -> Path:
    return Path(ispd15_dir) / Path(archive.drive_relpath).relative_to("CircuitNet-ISPD15")


def download_with_gdown(archive: Archive, dest_dir) -> Path:
    """Fallback when the Drive shortcut is unavailable: public-link download."""
    dest = Path(dest_dir) / archive.filename
    if dest.exists() and dest.stat().st_size == archive.size_bytes:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["gdown", f"https://drive.google.com/uc?id={archive.file_id}", "-O", str(dest)]
    subprocess.run(cmd, check=True)
    if dest.stat().st_size != archive.size_bytes:
        raise IOError(f"{dest}: size {dest.stat().st_size} != expected {archive.size_bytes}")
    return dest


# -------------------------------------------------------------------- extraction
def extract_members(archive_path, out_dir, prefixes=None, sample_names=None, log_every=500) -> int:
    """Stream-extract a .tar.gz, keeping only members under `prefixes` and (optionally)
    whose basename is one of `sample_names`. Uses tarfile's 'data' filter, so
    absolute paths, '..' and links are rejected. Returns the number of files written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(sample_names) if sample_names else None
    n = 0
    with tarfile.open(archive_path, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if prefixes and not any(member.name.startswith(p.rstrip("/") + "/") for p in prefixes):
                continue
            base = os.path.basename(member.name)
            if wanted is not None and base not in wanted and base.removesuffix(".npy") not in wanted:
                continue
            target = out_dir / member.name
            if target.exists() and target.stat().st_size == member.size:
                continue
            tar.extract(member, out_dir, filter="data")
            n += 1
            if log_every and n % log_every == 0:
                print(f"  extracted {n} files ...")
    return n


def copy_to_local(src, dst_dir) -> Path:
    """Copy one archive from Drive-FUSE to local disk (faster sequential reads)."""
    dst = Path(dst_dir) / Path(src).name
    if dst.exists() and dst.stat().st_size == Path(src).stat().st_size:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


# ------------------------------------------------------------------- raw layout
@dataclass(frozen=True)
class RawLayout:
    """Where extracted raw files live (Colab: /content/circuitnet_ispd15)."""
    root: Path

    def graph_info(self, design: str, kind: str) -> Path:
        return self.root / "graph_infromation" / f"{kind}_attr" / f"{design}_{kind}_attr.npy"

    def placement_gcell(self, sample: str) -> Path:
        return self.root / "instance_placement_gcell" / sample

    def placement_micron(self, sample: str) -> Path:
        return self.root / "instance_placement_micron" / f"{sample}.npy"

    def routability(self, key: str, sample: str) -> Path:
        return self.root / ROUTABILITY_MEMBERS[key] / sample

    def samples(self) -> list:
        d = self.root / "instance_placement_gcell"
        return sorted(p.name for p in d.iterdir()) if d.exists() else []


# Phase-1 real fixture samples (docs/PHASE_1_REPORT.md): three designs, so tests/test_real_data.py
# can build a train / val / test split from them.
REAL_FIXTURE_SAMPLES = ("68-mgc_fft_a-util_0.7", "3-mgc_pci_bridge32_a-util_0.85", "380-mgc_matrix_mult_a-util_0.75")


def subset_raw_tree(raw_root, out_root, samples=REAL_FIXTURE_SAMPLES) -> Path:
    """A RawLayout tree with only `samples`, made of symlinks into an extracted raw tree (nothing
    is copied or modified). For running tests/test_real_data.py on Colab:
    VERIHGN_FIXTURES=<out_root>. Raises if any needed file is missing."""
    src, dst = RawLayout(Path(raw_root)), RawLayout(Path(out_root))
    designs = sorted({parse_sample_name(s)["design"] for s in samples})
    pairs = [(src.graph_info(d, k), dst.graph_info(d, k)) for d in designs for k in ("node", "net", "pin")]
    optional = ("rudy", "rudy_pin", "macro_map")     # code-feature maps; test_real_data.py handles their absence
    for s in samples:
        pairs += [(src.placement_gcell(s), dst.placement_gcell(s)), (src.placement_micron(s), dst.placement_micron(s))]
        pairs += [(src.routability(k, s), dst.routability(k, s)) for k in ROUTABILITY_MEMBERS
                  if k not in optional or src.routability(k, s).exists()]
    missing = [str(a) for a, _ in pairs if not a.exists()]
    if missing:
        raise FileNotFoundError(f"raw files missing for the fixture subset: {missing[:5]}")
    for a, b in pairs:
        b.parent.mkdir(parents=True, exist_ok=True)
        if not b.exists():
            b.symlink_to(a.resolve(), target_is_directory=a.is_dir())
    return Path(out_root)
