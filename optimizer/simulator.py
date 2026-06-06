"""
simulator.py
============
ngspice interface for the SKY130 Schmitt Trigger.
Runs DC sweeps (up + down) and extracts V_PH and V_PL
from the transfer curve via linear interpolation.
"""

import subprocess, os, textwrap
import numpy as np

LIB     = "/home/nithin/.ciel/sky130A/libs.tech/combined/sky130.lib.spice"
TMP     = "tmp_bo.spice"
VDD     = 1.8
TIMEOUT = 35

# Original netlist nominal
NOMINAL = dict(W1=1.0, L1=2.5,  M1=1,
               W2=5.0, L2=2.5,  M2=1,
               W3=6.5, L3=0.15, M3=10,
               W4=16., L4=0.15, M4=10,
               W6=1.0, L6=0.15, M6=1)

# Parameter bounds for optimization [lo, hi]
PARAM_BOUNDS = {
    'W1': (0.42, 5.0),   'L1': (0.15, 8.0),
    'W2': (0.42, 12.0),  'L2': (0.15, 8.0),
    'W3': (0.42, 10.0),  'L3': (0.15, 3.0),
    'W4': (2.0,  40.0),  'L4': (0.15, 1.0),
    'W6': (0.42, 8.0),   'L6': (0.15, 12.0),
}
# Integer params kept fixed (M3=10, M4 swept separately)
PARAM_KEYS = list(PARAM_BOUNDS.keys())   # 10 continuous params


def _write_netlist(p: dict, vin_start: float, vin_stop: float, raw_path: str):
    def nmos(nm, d, g, s, b, W, L, M):
        ad, pd = W*0.29, W*2+0.58
        return (f"{nm} {d} {g} {s} {b} sky130_fd_pr__nfet_01v8 "
                f"L={L:.4f} W={W:.4f} nf=1 ad={ad:.4f} as={ad:.4f} "
                f"pd={pd:.4f} ps={pd:.4f} nrd={0.29/W:.5f} nrs={0.29/W:.5f} "
                f"sa=0 sb=0 sd=0 mult={M} m={M}")

    def pmos(nm, d, g, s, b, W, L, M):
        ad, pd = W*0.29, W*2+0.58
        return (f"{nm} {d} {g} {s} {b} sky130_fd_pr__pfet_01v8 "
                f"L={L:.4f} W={W:.4f} nf=1 ad={ad:.4f} as={ad:.4f} "
                f"pd={pd:.4f} ps={pd:.4f} nrd={0.29/W:.5f} nrs={0.29/W:.5f} "
                f"sa=0 sb=0 sd=0 mult={M} m={M}")

    step = 0.003 if vin_stop > vin_start else -0.003
    lines = [
        "** BO optimizer netlist",
        f".lib {LIB} tt", ".option wnflag=1", ".temp 27",
        f"VVDD VDD 0 {VDD}", f"VIN VIN 0 DC {vin_start}", "",
        nmos("XM1","net1","VIN", "0",   "GND", p['W1'],p['L1'],p.get('M1',1)),
        nmos("XM2","VOUT","VIN", "net1","GND", p['W2'],p['L2'],p.get('M2',1)),
        nmos("XM3","VDD", "VOUT","net1","GND", p['W3'],p['L3'],p.get('M3',10)),
        pmos("XM4","VOUT","VIN", "net2","VDD", p['W4'],p['L4'],p.get('M4',10)),
        pmos("XM5","net2","VIN", "VDD", "VDD", p['W4'],p['L4'],p.get('M4',10)),
        pmos("XM6","net2","VOUT","0",   "VDD", p['W6'],p['L6'],p.get('M6',1)),
        "", f".dc VIN {vin_start} {vin_stop} {step}",
        ".control", "set filetype=ascii", "run",
        f"write {raw_path} v(VIN) v(VOUT)", ".endc",
        ".GLOBAL VDD", ".end",
    ]
    with open(TMP, 'w') as f:
        f.write('\n'.join(lines))


def _parse_raw(raw_path: str):
    try:
        txt = open(raw_path, 'r', errors='replace').read()
    except FileNotFoundError:
        return None, None

    lines = txt.splitlines()
    vars_, ds, nv, np_ = [], 0, 0, 0
    i = 0
    while i < len(lines):
        l = lines[i].strip(); ll = l.lower()
        if ll.startswith('no. variables:'): nv = int(l.split(':')[1])
        elif ll.startswith('no. points:'): np_ = int(l.split(':')[1])
        elif ll.startswith('variables:'):
            i += 1
            while i < len(lines) and lines[i].startswith('\t'):
                pt = lines[i].strip().split()
                if len(pt) >= 2: vars_.append(pt[1].lower())
                i += 1
            continue
        elif ll.startswith('values:'): ds = i+1; break
        i += 1

    if not vars_ or np_ == 0 or ds == 0:
        return None, None

    nums = []
    for line in lines[ds:]:
        for t in line.split():
            try: nums.append(float(t))
            except: pass

    st = nv + 1
    if len(nums) < st: return None, None
    np_ = min(np_, len(nums)//st)
    D = np.array(nums[:st*np_]).reshape(np_, st)

    try:
        ci = vars_.index('v(vin)') + 1
        co = vars_.index('v(vout)') + 1
    except ValueError:
        ci, co = 1, 2

    return D[:, ci], D[:, co]


def _find_crossing(vin, vout, level=0.9, edge='fall'):
    for i in range(len(vout)-1):
        v0, v1 = vout[i], vout[i+1]
        if edge == 'fall' and v0 >= level > v1:
            frac = (level-v0) / (v1-v0+1e-15)
            return float(vin[i] + frac*(vin[i+1]-vin[i]))
        if edge == 'rise' and v0 <= level < v1:
            frac = (level-v0) / (v1-v0+1e-15)
            return float(vin[i] + frac*(vin[i+1]-vin[i]))
    return None


def simulate(p: dict) -> tuple:
    """
    Run ngspice DC sweeps and return (V_PH, V_PL).
    Returns (None, None) on failure.

    Parameters
    ----------
    p : dict  with keys matching PARAM_KEYS + M1/M2/M3/M4/M6

    Returns
    -------
    vph : float  upper switching threshold (V)
    vpl : float  lower switching threshold (V)
    """
    vph = vpl = None

    for vs, ve, suf, edge in [(0, VDD, 'u', 'fall'), (VDD, 0, 'd', 'rise')]:
        raw = f"tmp_bo_{suf}.raw"
        _write_netlist(p, vs, ve, raw)
        try:
            proc = subprocess.Popen(
                ["ngspice", "-b", TMP],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            proc.communicate(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            return None, None
        finally:
            if os.path.exists(TMP): os.remove(TMP)

        vin_a, vout_a = _parse_raw(raw)
        if os.path.exists(raw): os.remove(raw)
        if vin_a is None: return None, None

        val = _find_crossing(vin_a, vout_a, 0.9, edge)
        if val is None: return None, None

        if edge == 'fall': vph = val
        else:              vpl = val

    if vph and vpl and 0.05 < vpl < vph < VDD-0.05:
        return vph, vpl
    return None, None


def params_to_vector(p: dict) -> np.ndarray:
    """Convert param dict → normalized [0,1]^10 vector."""
    x = []
    for k in PARAM_KEYS:
        lo, hi = PARAM_BOUNDS[k]
        x.append((p[k] - lo) / (hi - lo))
    return np.array(x)


def vector_to_params(x: np.ndarray, base: dict = None) -> dict:
    """Convert normalized [0,1]^10 vector → param dict."""
    p = dict(base or NOMINAL)
    for i, k in enumerate(PARAM_KEYS):
        lo, hi = PARAM_BOUNDS[k]
        p[k] = float(np.clip(lo + x[i]*(hi-lo), lo, hi))
    return p
