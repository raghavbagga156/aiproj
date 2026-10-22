"""Per-design netlist from CircuitNet `graph_information`.

Array semantics (verified in Phase 0 against the arrays and against CircuitNet's
own `build_graph_demo/build_graph.py`):
  node_attr: (2, 1+C)  row0 instance names, row1 master names; index 0 = top module
  net_attr:  (1, E)    net names
  pin_attr:  (3, P)    row0 pin name, row1 net index (int, or list of ints),
                       row2 node index into node_attr (>= 1)
Top-level IO ports are not part of graph_information (the DEF lists them under
PINS; they have no node here). Macro pins are named after the net they connect.
"""
import hashlib
from dataclasses import dataclass

import numpy as np

from ..utils.safe_npy import load_npy

# Standard-cell output pin names in the ISPD15 mgc library. Phase 1 audit of all
# 9 official-scope designs: 16 masters, every one has exactly one pin "o"; all
# other pins are a/b/c/d/ck (inputs).
STD_OUTPUT_PINS = frozenset({"o"})


@dataclass
class Netlist:
    design: str
    cell_names: np.ndarray      # (C,) str
    masters: np.ndarray         # (C,) str
    is_macro: np.ndarray        # (C,) bool
    num_nets: int
    pin_name: np.ndarray        # (P,) str
    pin_net: np.ndarray         # (P,) int64
    pin_cell: np.ndarray        # (P,) int64, 0-based cell index
    pin_is_output: np.ndarray   # (P,) float32 in {0, 1}
    macro_pin_direction: str    # how macro pin direction was set

    @property
    def num_cells(self) -> int:
        return len(self.cell_names)

    @property
    def num_pins(self) -> int:
        return len(self.pin_net)

    def signature(self) -> str:
        """Hash of the connectivity + masters, used to find netlist-identical designs."""
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(self.masters).astype("U").tobytes())
        h.update(self.pin_net.tobytes())
        h.update(self.pin_cell.tobytes())
        h.update(np.ascontiguousarray(self.pin_name).astype("U").tobytes())
        return h.hexdigest()


def macro_flags(names: np.ndarray, masters: np.ndarray, design: str) -> np.ndarray:
    """mgc designs name each macro instance after its own master (h0/h0, hh4/hh4, ...);
    no standard cell does. Superblue macros use distinct master names, so this
    rule is refused there (LEF CLASS BLOCK would be needed)."""
    if "superblue" in design:
        raise NotImplementedError(
            "macro detection for superblue designs needs LEF CLASS BLOCK; not implemented "
            "(outside the approved 9-design scope)")
    return names == masters


def load_netlist(paths, design: str, macro_pin_direction: str = "input") -> Netlist:
    node = load_npy(paths.graph_info(design, "node"))
    net = load_npy(paths.graph_info(design, "net"))
    pin = load_npy(paths.graph_info(design, "pin"))

    names = np.asarray(node[0][1:], dtype=str)
    masters = np.asarray(node[1][1:], dtype=str)
    num_nets = int(np.asarray(net).shape[1])

    pin_names, pin_nets, pin_nodes = [], [], []
    for pname, pnet, pnode in zip(pin[0], pin[1], pin[2]):
        for n in (pnet if isinstance(pnet, (list, tuple)) else [pnet]):
            pin_names.append(pname)
            pin_nets.append(int(n))
            pin_nodes.append(int(pnode))
    pin_name = np.asarray(pin_names, dtype=str)
    pin_net = np.asarray(pin_nets, dtype=np.int64)
    pin_cell = np.asarray(pin_nodes, dtype=np.int64) - 1

    if pin_cell.min() < 0 or pin_cell.max() >= len(names):
        raise ValueError(f"{design}: pin node index out of range")
    if pin_net.min() < 0 or pin_net.max() >= num_nets:
        raise ValueError(f"{design}: pin net index out of range")

    is_macro = macro_flags(names, masters, design)
    on_macro = is_macro[pin_cell]
    out = np.isin(pin_name, list(STD_OUTPUT_PINS)) & ~on_macro
    unknown_std = ~on_macro & ~out & ~np.isin(pin_name, ["a", "b", "c", "d", "ck"])
    if unknown_std.any():
        raise ValueError(f"{design}: unexpected standard-cell pin names "
                         f"{sorted(set(pin_name[unknown_std]))[:10]}")
    if macro_pin_direction == "input":
        pass
    elif macro_pin_direction == "output":
        out = out | on_macro
    else:
        raise ValueError(f"macro_pin_direction must be input|output, got {macro_pin_direction}")

    return Netlist(design, names, masters, is_macro, num_nets, pin_name, pin_net, pin_cell,
                   out.astype(np.float32), macro_pin_direction)
