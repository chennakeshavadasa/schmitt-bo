"""
simulator.py
============
ngspice interface for the SKY130 6T CMOS Schmitt Trigger.

Runs two DC sweeps per evaluation:
  Up-sweep   (VIN: 0 -> 1.8 V): VOUT falls  -> extracts V_PH
  Down-sweep (VIN: 1.8 -> 0 V): VOUT rises  -> extracts V_PL

Threshold extraction: linear interpolation of VOUT at the 0.9 V crossing.
Each sweep pair takes ~33-37 s (SKY130 BSIM4 model load dominates).

Edit LIB below to match your PDK installation before running.
"""

import subprocess
import os
import numpy as np


# ---------------------------------------------------------------------------
# Local configuration -- edit these paths
# ---------------------------------------------------------------------------
LIB     = "/home/nithin/.ciel/sky130A/libs.tech/combined/sky130.lib.spice"
TMP     = "tmp_bo.spice"       # temp netlist, written + deleted each sim
VDD     = 1.8                  # supply voltage (V)
TIMEOUT = 45                   # hard kill after N seconds (handles hangs)


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------
# 10 continuous sizing variables: W and L for each of 5 devices.
# XM4 and XM5 share the same W/L (tied in the topology).
# XM3 multiplicity (m=10) is fixed -- only W/L are optimized.

NOMINAL = dict(
    W1=1.0,  L1=2.5,    # XM1: NMOS series
    W2=5.0,  L2=2.5,    # XM2: NMOS main
    W3=6.5,  L3=0.15,   # XM3: NMOS feedback  (mult=10 fixed)
    W4=16.0, L4=0.15,   # XM4/XM5: PMOS main  (mult=10 fixed, tied)
    W6=1.0,  L6=0.15,   # XM6: PMOS feedback
)

PARAM_KEYS = ["W1", "L1", "W2", "L2", "W3", "L3", "W4", "L4", "W6", "L6"]

PARAM_BOUNDS = {
    "W1": (0.42,  4.0),   "L1": (0.15,  3.0),
    "W2": (0.42,  8.0),   "L2": (0.15,  3.0),
    "W3": (0.42,  6.5),   "L3": (0.15,  2.0),
    "W4": (4.0,  32.0),   "L4": (0.15,  0.8),
    "W6": (0.42,  6.0),   "L6": (0.15,  8.0),
}

# Real ngspice sensitivity (v8 scan, 58 sims, 27 C, tt corner) -- V/um
SENSITIVITY = {
    "L4": (-1.300, -1.782),   # dominant lever
    "W4": (+0.002, +0.045),   # lifts V_PL
    "W3": (-0.081,  0.000),   # shifts V_PH only
    "L6": (+0.003, +0.078),   # main V_PL lever
    "W6": (+0.003, -0.084),   # wider -> lower V_PL (counterintuitive)
    "W1": (+0.021, +0.012),   # minor
}


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def params_to_vector(p):
    """Convert raw-um parameter dict -> normalized [0, 1]^10 vector."""
    return np.array(
        [(p[k] - PARAM_BOUNDS[k][0]) / (PARAM_BOUNDS[k][1] - PARAM_BOUNDS[k][0])
         for k in PARAM_KEYS]
    )


def vector_to_params(x, base=None):
    """Convert normalized [0, 1]^10 vector -> raw-um parameter dict."""
    p = dict(base or NOMINAL)
    for i, k in enumerate(PARAM_KEYS):
        lo, hi = PARAM_BOUNDS[k]
        p[k] = float(np.clip(lo + x[i] * (hi - lo), lo, hi))
    return p


# ---------------------------------------------------------------------------
# Netlist generation
# ---------------------------------------------------------------------------

def _write_netlist(p, vin_start, vin_stop, raw_path, lib=LIB):
    """Write a parametrized SKY130 Schmitt Trigger netlist for one DC sweep."""

    def nmos(nm, d, g, s, b, W, L, M=1):
        ad, pd = W * 0.29, W * 2 + 0.58
        return (
            f"{nm} {d} {g} {s} {b} sky130_fd_pr__nfet_01v8 "
            f"L={L:.4f} W={W:.4f} nf=1 "
            f"ad={ad:.4f} as={ad:.4f} pd={pd:.4f} ps={pd:.4f} "
            f"nrd={0.29/W:.5f} nrs={0.29/W:.5f} "
            f"sa=0 sb=0 sd=0 mult={M} m={M}"
        )

    def pmos(nm, d, g, s, b, W, L, M=1):
        ad, pd = W * 0.29, W * 2 + 0.58
        return (
            f"{nm} {d} {g} {s} {b} sky130_fd_pr__pfet_01v8 "
            f"L={L:.4f} W={W:.4f} nf=1 "
            f"ad={ad:.4f} as={ad:.4f} pd={pd:.4f} ps={pd:.4f} "
            f"nrd={0.29/W:.5f} nrs={0.29/W:.5f} "
            f"sa=0 sb=0 sd=0 mult={M} m={M}"
        )

    step = 0.01 if vin_stop > vin_start else -0.003
    lines = [
        "** Bayesian Optimizer -- SKY130 Schmitt Trigger",
        f".lib {lib} tt",
        ".option wnflag=1",
        ".temp 27",
        f"VVDD VDD 0 {VDD}",
        f"VIN  VIN 0 DC {vin_start}",
        "",
        nmos("XM1", "net1", "VIN",  "0",    "GND", p["W1"], p["L1"], p.get("M1", 1)),
        nmos("XM2", "VOUT", "VIN",  "net1", "GND", p["W2"], p["L2"], p.get("M2", 1)),
        nmos("XM3", "VDD",  "VOUT", "net1", "GND", p["W3"], p["L3"], p.get("M3", 10)),
        pmos("XM4", "VOUT", "VIN",  "net2", "VDD", p["W4"], p["L4"], p.get("M4", 10)),
        pmos("XM5", "net2", "VIN",  "VDD",  "VDD", p["W4"], p["L4"], p.get("M4", 10)),
        pmos("XM6", "net2", "VOUT", "0",    "VDD", p["W6"], p["L6"], p.get("M6", 1)),
        "",
        f".dc VIN {vin_start} {vin_stop} {step}",
        ".control",
        "set filetype=ascii",
        "run",
        f"write {raw_path} v(VIN) v(VOUT)",
        ".endc",
        ".GLOBAL VDD",
        ".end",
    ]
    with open(TMP, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Raw file parser
# ---------------------------------------------------------------------------

def _parse_raw(raw_path):
    """
    Parse ngspice ASCII .raw output.
    Returns (vin_array, vout_array) or (None, None) on any parse failure.
    """
    try:
        txt = open(raw_path, errors="replace").read()
    except FileNotFoundError:
        return None, None

    lines = txt.splitlines()
    vars_, data_start, n_vars, n_pts = [], 0, 0, 0
    i = 0
    while i < len(lines):
        ll = lines[i].strip().lower()
        if ll.startswith("no. variables:"):
            n_vars = int(lines[i].split(":")[1])
        elif ll.startswith("no. points:"):
            n_pts = int(lines[i].split(":")[1])
        elif ll.startswith("variables:"):
            i += 1
            while i < len(lines) and lines[i].startswith("\t"):
                parts = lines[i].strip().split()
                if len(parts) >= 2:
                    vars_.append(parts[1].lower())
                i += 1
            continue
        elif ll.startswith("values:"):
            data_start = i + 1
            break
        i += 1

    if not vars_ or n_pts == 0 or data_start == 0:
        return None, None

    nums = []
    for line in lines[data_start:]:
        for tok in line.split():
            try:
                nums.append(float(tok))
            except ValueError:
                pass

    stride = n_vars + 1
    n_pts = min(n_pts, len(nums) // stride)
    if n_pts == 0:
        return None, None

    D = np.array(nums[: stride * n_pts]).reshape(n_pts, stride)
    try:
        ci = vars_.index("v(vin)") + 1
        co = vars_.index("v(vout)") + 1
    except ValueError:
        ci, co = 1, 2

    return D[:, ci], D[:, co]


# ---------------------------------------------------------------------------
# Threshold extraction
# ---------------------------------------------------------------------------

def _find_crossing(vin, vout, level=0.9, edge="fall"):
    """
    Find VIN where VOUT crosses `level` on the given edge.
    Linear interpolation between the two bracketing samples.

    edge='fall': VOUT transitions high->low  (gives V_PH)
    edge='rise': VOUT transitions low->high  (gives V_PL)
    """
    for i in range(len(vout) - 1):
        v0, v1 = vout[i], vout[i + 1]
        if edge == "fall" and v0 >= level > v1:
            frac = (level - v0) / (v1 - v0 + 1e-15)
            return float(vin[i] + frac * (vin[i + 1] - vin[i]))
        if edge == "rise" and v0 <= level < v1:
            frac = (level - v0) / (v1 - v0 + 1e-15)
            return float(vin[i] + frac * (vin[i + 1] - vin[i]))
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate(p):
    """
    Run ngspice DC sweeps and return (V_PH, V_PL) in volts.

    Parameters
    ----------
    p : dict
        Parameter dict with keys from PARAM_KEYS (W/L values in um).

    Returns
    -------
    (vph, vpl) : (float, float) or (None, None) on any failure.

    Failure cases handled:
        - ngspice not on PATH          -> FileNotFoundError caught silently
        - BSIM4 non-convergence hang   -> process killed after TIMEOUT s
        - Missing or corrupt .raw file -> _parse_raw returns (None, None)
        - VOUT never crosses 0.9 V     -> _find_crossing returns None
        - Physically invalid result    -> sanity check at end
    """
    vph = vpl = None

    for vin_s, vin_e, suf, edge in [
        (0,   VDD, "u", "fall"),
        (VDD, 0,   "d", "rise"),
    ]:
        raw = f"tmp_bo_{suf}.raw"
        _write_netlist(p, vin_s, vin_e, raw)

        try:
            proc = subprocess.Popen(
                ["ngspice", "-b", TMP],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                proc.communicate(timeout=TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return None, None
        except FileNotFoundError:
            return None, None
        finally:
            if os.path.exists(TMP):
                os.remove(TMP)

        vin_arr, vout_arr = _parse_raw(raw)
        if os.path.exists(raw):
            os.remove(raw)
        if vin_arr is None:
            return None, None

        val = _find_crossing(vin_arr, vout_arr, level=0.9, edge=edge)
        if val is None:
            return None, None

        if edge == "fall":
            vph = val
        else:
            vpl = val

    if vph and vpl and 0.05 < vpl < vph < VDD - 0.05:
        return vph, vpl
    return None, None
