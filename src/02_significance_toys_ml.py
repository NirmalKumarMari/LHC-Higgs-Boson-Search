"""Labs 7 and 8: asymptotic significance, Poisson toy Monte Carlo, ML selection.

Originally a Colab notebook. Needs higgs_100000_<SAMPLE>.pkl, qcd_100000_<SAMPLE>.pkl,
and data_{high,low}Lumi_<SAMPLE>.h5 in the working directory (not included).
"""

# %%
# REQUIRED LIBRARIES & DATASET INITIALIZATION

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

plt.rcParams["figure.dpi"] = 120

# ---- User file transfer ----
# Data files are expected in DATA_DIR (see README). Colab upload removed.

# ---- Utility functions for reading data ----
def load_pkl(filename):
    return pd.read_pickle(filename)

def load_pseudo_data(filename):
    with h5py.File(filename, "r") as f:
        data_group = f["data"]
        columns = [
            x.decode("utf-8") if isinstance(x, bytes) else x
            for x in data_group["block0_items"][:]
        ]
        data_values = data_group["block0_values"][:]
    return pd.DataFrame(data_values, columns=columns)

# ---- Select analysis region ----
SAMPLE = "pt_250_500"   # change this if needed

# Dynamically construct file paths
higgs_file = f"higgs_100000_{SAMPLE}.pkl"
qcd_file   = f"qcd_100000_{SAMPLE}.pkl"
pseudo_file = f"data_highLumi_{SAMPLE}.h5"

# ---- Import experimental samples ----
h = load_pkl(higgs_file)
q = load_pkl(qcd_file)

h["__label__"] = 1  # tag Higgs events
q["__label__"] = 0  # tag QCD events

df_train = pd.concat([h, q], ignore_index=True)  # merge datasets

# ---- Standard production rates ----
EXPECTED = {
    "pt_250_500":  {"N_higgs": 100, "N_qcd": 20000},
    "pt_1000_1200": {"N_higgs": 50,  "N_qcd": 2000},
}

N_higgs_nom = EXPECTED[SAMPLE]["N_higgs"]  # nominal Higgs count
N_qcd_nom   = EXPECTED[SAMPLE]["N_qcd"]

print("Imported:", len(h), "signal,", len(q), "background")
print("Projected yields → Higgs:", N_higgs_nom, " QCD:", N_qcd_nom)

# ---- Normalization factors ----
signal_weight     = N_higgs_nom / len(h)
background_weight = N_qcd_nom   / len(q)

print("Signal weight     =", signal_weight)
print("Background scaling =", background_weight)

# ---- Retrieve simulated observations (Lab 8) ----
data_df = load_pseudo_data(pseudo_file)  # synthetic measurement
print("Simulation size:", data_df.shape)

# %%
# Lab 5 -  BASIC PLOTS

# Invariant mass spectrum
plt.figure(figsize=(8,5))
plt.hist(q["mass"], bins=50, alpha=0.5, density=True, label="QCD")
plt.hist(h["mass"], bins=50, alpha=0.5, density=True, label="Higgs")
plt.xlabel("Invariant mass (GeV)")
plt.ylabel("Frequency (normalized)")
plt.title("Invariant mass: Higgs signal vs QCD noise")
plt.legend()
plt.grid()
plt.show()

# d2 substructure variable
plt.figure(figsize=(8,5))
plt.hist(q["d2"], bins=50, alpha=0.5, density=True, label="QCD")
plt.hist(h["d2"], bins=50, alpha=0.5, density=True, label="Higgs")
plt.xlabel("d2 (substructure)")
plt.ylabel("Frequency (normalized)")
plt.title("d2 substructure: Higgs signal vs QCD noise")
plt.legend()
plt.grid()
plt.show()

# Particle momentum features
features = ["pt", "eta", "phi"]  # momentum, pseudorapidity, azimuth
plt.figure(figsize=(12,8))
for i, feat in enumerate(features):
    plt.subplot(2,2,i+1)
    plt.hist(q[feat], bins=50, alpha=0.5, density=True, label="QCD")
    plt.hist(h[feat], bins=50, alpha=0.5, density=True, label="Higgs")
    plt.xlabel(feat)
    plt.ylabel("Frequency (normalized)")
    plt.title(f"{feat} spectrum")
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.show()

# %%
# BIDIMENSIONAL CORRELATIONS

plt.figure(figsize=(7,5))
plt.scatter(h["mass"], h["d2"], s=5, alpha=0.3, label="Higgs")
plt.xlabel("Invariant mass (GeV)")
plt.ylabel("d2")
plt.title("Higgs signal: mass-d2 correlation")
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(7,5))
plt.scatter(q["mass"], q["d2"], s=5, alpha=0.3, label="QCD")
plt.xlabel("Invariant mass (GeV)")
plt.ylabel("d2")
plt.title("QCD noise: mass-d2 correlation")
plt.grid()
plt.legend()
plt.show()

# %%
# SCALED HISTOGRAMS

plt.figure(figsize=(8,5))
plt.hist(
    [q["mass"], h["mass"]],
    bins=50,
    weights=[
        np.full(len(q), background_weight),
        np.full(len(h), signal_weight)
    ],
    stacked=True,
    label=["QCD", "Higgs"]
)
plt.xlabel("Invariant mass (GeV)")
plt.ylabel("Scaled counts")
plt.title("Scaled invariant mass distribution")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8,5))
plt.hist(
    [q["d2"], h["d2"]],
    bins=50,
    weights=[
        np.full(len(q), background_weight),
        np.full(len(h), signal_weight)
    ],
    stacked=True,
    label=["QCD", "Higgs"]
)
plt.xlabel("d2 (substructure)")
plt.ylabel("Scaled counts")
plt.title("Scaled d2 distribution")
plt.legend()
plt.grid()
plt.show()

# %%
# MASS WINDOW + d2 OPTIMIZATION

mass_min, mass_max = 120, 130  # signal region window

sig_mass_mask = (h["mass"] >= mass_min) & (h["mass"] <= mass_max)
bkg_mass_mask = (q["mass"] >= mass_min) & (q["mass"] <= mass_max)

d2_sig = h.loc[sig_mass_mask, "d2"]
d2_bkg = q.loc[bkg_mass_mask, "d2"]

d2_scan = np.linspace(0, 10, 120)
Zvals = []

for cut in d2_scan:
    S = (d2_sig <= cut).sum() * signal_weight
    B = (d2_bkg <= cut).sum() * background_weight
    Z = S / np.sqrt(B) if B > 0 else 0
    Zvals.append(Z)

Zvals = np.array(Zvals)

best_idx = np.argmax(Zvals)
best_d2 = d2_scan[best_idx]

print("Optimal d2 cutoff =", best_d2)
print("Peak significance Z =", Zvals[best_idx])

plt.figure(figsize=(8,5))
plt.plot(d2_scan, Zvals)
plt.xlabel("d2 threshold")
plt.ylabel("Z statistic = signal / sqrt(noise)")
plt.title("Statistical significance vs d2 threshold")
plt.grid()
plt.show()

# %%
# FINAL MASS DISTRIBUTION AFTER BOTH CUTS

# Apply final selections
sig_final = sig_mass_mask & (h["d2"] <= best_d2)
bkg_final = bkg_mass_mask & (q["d2"] <= best_d2)

# Extract the selected mass values
mass_sig_final = h["mass"][sig_final]
mass_bkg_final = q["mass"][bkg_final]

# Build weights that MATCH the length of each mass array
weights_sig_final = np.full(len(mass_sig_final), signal_weight, dtype=float)
weights_bkg_final = np.full(len(mass_bkg_final), background_weight, dtype=float)

# Compute final significance
S_final = len(mass_sig_final) * signal_weight
B_final = len(mass_bkg_final) * background_weight
Z_final = S_final / np.sqrt(B_final) if B_final > 0 else 0

print("Final weighted S =", S_final)
print("Final weighted B =", B_final)
print("Final significance =", Z_final)

# Final weighted histogram
plt.figure(figsize=(8,5))
plt.hist(
    [mass_bkg_final, mass_sig_final],
    bins=50,
    stacked=True,
    weights=[weights_bkg_final, weights_sig_final],
    label=["QCD Background", "Higgs Signal"]
)
plt.xlabel("Invariant mass (GeV)")
plt.ylabel("Weighted Counts")
plt.title("Final Mass Distribution after Optimal Mass + d2 Cuts")
plt.legend()
plt.grid()
plt.show()

# %%
# Helpers for Lab 7 (mass, d2, t21 selection + significance)

import numpy as np
import matplotlib.pyplot as plt

# event weights based on expected nominal yields
w_sig = N_higgs_nom / len(h)
w_bkg = N_qcd_nom / len(q)

print("Signal weight per event =", w_sig)
print("Background weight per event =", w_bkg)

# Apply rectangular cuts in (mass, d2, t21)
def apply_selection(mass_low=120, mass_high=130,
                    d2_max=2.6, t21_max=0.4):
    sig_sel = (
        (h["mass"] >= mass_low) & (h["mass"] <= mass_high) &
        (h["d2"]  <= d2_max) &
        (h["t21"] <= t21_max)
    )
    bkg_sel = (
        (q["mass"] >= mass_low) & (q["mass"] <= mass_high) &
        (q["d2"]  <= d2_max) &
        (q["t21"] <= t21_max)
    )

    # expected yields for nominal luminosity
    s = sig_sel.sum() * w_sig
    b = bkg_sel.sum() * w_bkg
    return sig_sel, bkg_sel, s, b

# Lab 5 simple approximation
def Z_simple(s, b):
    if b <= 0:
        return 0.0
    return s / np.sqrt(b)

# Lab 7 "asymptotic" (Cowan eq. 97) – vectorized
def Z_asymptotic(s, b):
    s = np.asarray(s, dtype=float)
    b = np.asarray(b, dtype=float)

    Z = np.zeros_like(s, dtype=float)
    mask = (s > 0) & (b > 0)

    sb = s[mask] + b[mask]
    term = sb * np.log(1.0 + s[mask]/b[mask]) - s[mask]
    Z[mask] = np.sqrt(2.0 * term)

    return Z

# %%
# Three-dimensional parameter sweep: mass, d2, t21


mass_low, mass_high = 120, 130   # reuse Lab 5 mass interval


d2_grid  = np.linspace(0.5, 5.0, 30)
t21_grid = np.linspace(0.1, 0.8, 30)


best = {  # container for top result
    "Z_asym": 0.0,  # best exact significance
    "Z_simple": 0.0,  # best basic significance
    "d2_max": None,  # optimal d2 boundary
    "t21_max": None,  # optimal t21 boundary
    "s": None,  # winning signal count
    "b": None,  # winning noise count
    "sig_mask": None,  # winning Higgs filter
    "bkg_mask": None,  # winning QCD filter
}


for d2_max in d2_grid:
    for t21_max in t21_grid:
        sig_sel, bkg_sel, s, b = apply_selection(
            mass_low=mass_low,
            mass_high=mass_high,
            d2_max=d2_max,
            t21_max=t21_max
        )


        Z_a = float(Z_asymptotic(s, b))
        Z_s = Z_simple(s, b)


        if Z_a > best["Z_asym"]:
            best.update({
                "Z_asym": Z_a,
                "Z_simple": Z_s,
                "d2_max": d2_max,
                "t21_max": t21_max,
                "s": s,
                "b": b,
                "sig_mask": sig_sel,
                "bkg_mask": bkg_sel,
            })


print("Optimal three-dimensional selection (mass, d2, t21):")
print(f"  mass range : {mass_low}–{mass_high} GeV")
print(f"  d2 ≤ {best['d2_max']:.3f}")
print(f"  t21 ≤ {best['t21_max']:.3f}")
print(f"  projected S = {best['s']:.2f}")
print(f"  projected B = {best['b']:.2f}")
print(f"  Z_basic       (s/√b)   = {best['Z_simple']:.2f}")
print(f"  Z_exact      (Cowan)   = {best['Z_asym']:.2f}")

# %%
# Statistical power against cumulative yield and two-dimensional visualization


S0 = best["s"]  # baseline signal
B0 = best["b"]  # baseline noise
print(f"Foundation post-optimization: S0 = {S0:.2f}, B0 = {B0:.2f}")


# 1D: modify aggregate, maintain S/B ratio (intensity scaling)
scales = np.linspace(0.1, 5.0, 50)   # 0.1x through 5x reference
Z_vs_tot = []
Ntot_vals = []


for k in scales:
    s = k * S0
    b = k * B0
    z = float(Z_asymptotic(s, b))
    Z_vs_tot.append(z)
    Ntot_vals.append(s + b)


Z_vs_tot = np.array(Z_vs_tot)
Ntot_vals = np.array(Ntot_vals)


plt.figure(figsize=(7,5))
plt.plot(Ntot_vals, Z_vs_tot, marker="o", linestyle="-")
plt.axhline(5.0, linestyle="--", label="5σ")
plt.xlabel("Total projected particles S + B")
plt.ylabel("Exact central statistical power Z")
plt.title("Projected statistical significance vs total production (constant S/B)")
plt.legend()
plt.grid(True)
plt.show()


# Identify when Z dips below 5σ threshold
below5 = np.where(Z_vs_tot < 5.0)[0]
if len(below5):
    idx = below5[-1]
    print(f"Z dips below 5σ when S+B ≈ {Ntot_vals[idx]:.1f}, "
          f"S ≈ {scales[idx]*S0:.1f}, B ≈ {scales[idx]*B0:.1f}")
else:
    print("Z remains above 5σ across this range — widen 'scales' parameter to observe descent below 5σ.")


# 2D: lattice across (S, B) with hue encoding Z
S_grid = np.linspace(0, 3*S0, 40)
B_grid = np.linspace(0.1, 3*B0, 40)  # begin at 0.1 to prevent b=0 singularity
S_mesh, B_mesh = np.meshgrid(S_grid, B_grid)
Z_mesh = Z_asymptotic(S_mesh, B_mesh)


plt.figure(figsize=(7,5))
pcm = plt.pcolormesh(
    S_mesh + B_mesh,  # x-axis: signal + noise aggregate
    S_mesh,           # y-axis: signal yield
    Z_mesh,
    shading="auto"
)
plt.colorbar(pcm, label="Projected statistical power Z")
plt.xlabel("Signal + Noise particle count")
plt.ylabel("Signal particle count")
plt.title("Projected power distribution in (aggregate, signal) plane")
plt.show()

# %%
N_higgs_exp = 100    # baseline signal volume pre-filter
N_qcd_exp   = 20000  # baseline noise volume pre-filter


# High-momentum variant (deactivate above, enable this if preferred):
# N_higgs_exp = 50
# N_qcd_exp   = 2000


# ---- ACQUISITION PASS-RATES (from Lab 7) ----
eps_s = 0.14
eps_b = 0.00002


# Post-filter yields (at standard beam intensity)
s0 = eps_s * N_higgs_exp
b0 = eps_b * N_qcd_exp


# Lattice configuration options
# Adjust based on your experimental requirements
n_higgs_max = 3 * s0       # triple reference post-filter signal
n_tot_max   = 3 * (s0+b0)  # triple reference post-filter aggregate


n_higgs_vals = np.linspace(0.0, n_higgs_max, 80)
n_tot_vals_2d = np.linspace(1.0, n_tot_max, 120)  # start beyond zero to prevent background singularity


N_tot_grid, S_grid = np.meshgrid(n_tot_vals_2d, n_higgs_vals)


# Noise calculation: aggregate minus signal; enforce non-negative
B_grid = N_tot_grid - S_grid
valid = B_grid >= 0  # mask for physical region


Z_grid = np.full_like(S_grid, np.nan, dtype=float)
Z_grid[valid] = Z_asymptotic(S_grid[valid], B_grid[valid])


plt.figure(figsize=(8,6))
# Render heatmap with proper coordinate bounds
im = plt.imshow(
    Z_grid,
    origin='lower',
    extent=[
        n_tot_vals_2d.min(), n_tot_vals_2d.max(),
        n_higgs_vals.min(), n_higgs_vals.max()
    ],
    aspect='auto'
)
cbar = plt.colorbar(im)
cbar.set_label("Projected central statistical power Z")


plt.xlabel(r"$n_{\rm tot} = s + b$ (post-filter)")
plt.ylabel(r"$n_{\rm higgs} = s$ (post-filter)")
plt.title(r"Power distribution across $(n_{\rm tot}, n_{\rm higgs})$ domain")


# Supplementary: sketch isocontours at reference Z marks (e.g. 2σ, 3σ, 5σ)
Z_levels = [2.0, 3.0, 5.0]
CS = plt.contour(
    N_tot_grid,
    S_grid,
    Z_grid,
    levels=Z_levels,
    colors='white',
    linewidths=1.0
)
plt.clabel(CS, inline=True, fontsize=8, fmt="%.0fσ")


plt.tight_layout()
plt.show()

# %%
sig_sel, bkg_sel, s, b = apply_selection(
    mass_low=mass_low,
    mass_high=mass_high,
    d2_max=best['d2_max'],
    t21_max=best['t21_max']
)


s_eff = np.mean(sig_sel)
b_eff = np.mean(bkg_sel)


print(f'signal pass-rate = {s_eff:.3f}')
print(f'background pass-rate = {b_eff:.7f}')

# %%
from scipy.stats import poisson, norm
import scipy.stats as stats


def Z_from_poisson(n_obs, b_exp):
    """
    Precise discovery statistical power for count-based measurement.
    n_obs: sequence of recorded aggregate counts per replica (signal + noise).
    b_exp: anticipated noise (scalar or vectorizable to n_obs shape).


    Delivers Z >= 0 (we enforce Z=0 when n_obs <= b_exp, indicating no upward deviation).
    """
    n_obs = np.asarray(n_obs, dtype=int)
    b_exp = np.asarray(b_exp, dtype=float)


    # Poisson survival: P(N >= n_obs | mean = b_exp)
    # sf(k-1) delivers P(N >= k)
    p_vals = stats.poisson.sf(n_obs - 1, mu=b_exp)


    # Bilateral Gaussian equivalent power
    Z = stats.norm.isf(p_vals)


    # Discovery protocol: set Z=0 when no surplus observed
    Z[n_obs <= b_exp] = 0.0


    return Z

# %%
rng = np.random.default_rng(seed=12345)  # stability guarantee


def run_toys(n_s, n_b, n_toys=10000):
    """
    Execute stochastic replicas for specified anticipated signal and noise yields (post-filter).


    n_s, n_b: anticipated means AFTER filtering (scalar).
    n_toys:   quantity of pseudo-replicas.


    Yields:
      Z_toys: array of precise powers (one per replica).
    """
    # Harvest post-selection signal and noise from Poisson ensemble
    s_toys = rng.poisson(lam=n_s, size=n_toys)
    b_toys = rng.poisson(lam=n_b, size=n_toys)


    n_obs = s_toys + b_toys  # total detected post-selection


    # Precise discovery power for noise-only foundation
    Z_toys = Z_from_poisson(n_obs, b_exp=n_b)


    return Z_toys

# %%
# Brightness scaling multipliers
k_values = np.linspace(0.05, 5.0, 40)  # reduced count for computational feasibility


Z_as_curve = []
Z_toy_median_curve = []
N_tot_curve = []


n_toys_1d = 5000  # boost if additional resolution sought


for k in k_values:
    n_s = k * s0
    n_b = k * b0
    N_tot = n_s + n_b


    # Analytical Asimov power
    Z_as = Z_asymptotic(n_s, n_b)


    # Replica power distribution
    Z_toys = run_toys(n_s, n_b, n_toys=n_toys_1d)
    Z_med = np.median(Z_toys)


    Z_as_curve.append(Z_as)
    Z_toy_median_curve.append(Z_med)
    N_tot_curve.append(N_tot)


Z_as_curve = np.array(Z_as_curve)
Z_toy_median_curve = np.array(Z_toy_median_curve)
N_tot_curve = np.array(N_tot_curve)


# Visualization
plt.figure(figsize=(7,5))
plt.plot(N_tot_curve, Z_as_curve, label="Asimov $Z_A$ (theoretical)")
plt.plot(N_tot_curve, Z_toy_median_curve, 'o-', label="Replica sampling midpoint Z")


plt.xlabel(r"Total projected yield $N_{\rm tot} = s + b$ (post-filter)")
plt.ylabel("Median discovery statistical power Z")
plt.title("Replica validation of theoretical significance vs total production")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Mesh geometry for sampling
n_higgs_max = 3 * s0
n_tot_max   = 3 * (s0 + b0)


n_higgs_vals = np.linspace(0.0, n_higgs_max, 40)
n_tot_vals_2d = np.linspace(1.0, n_tot_max, 40)  # start at 1


N_tot_grid, S_grid = np.meshgrid(n_tot_vals_2d, n_higgs_vals)
B_grid = N_tot_grid - S_grid


Z_med_grid = np.full_like(S_grid, np.nan, dtype=float)


n_toys_2d = 2000  # modifiable per cell


for i in range(S_grid.shape[0]):
    for j in range(S_grid.shape[1]):
        n_s = S_grid[i, j]
        n_b = B_grid[i, j]


        if n_b < 0:
            continue  # non-physical zone; stay NaN
        if n_s < 0:
            continue


        Z_toys = run_toys(n_s, n_b, n_toys=n_toys_2d)
        Z_med_grid[i, j] = np.median(Z_toys)


plt.figure(figsize=(8,6))
im = plt.imshow(
    Z_med_grid,
    origin='lower',
    extent=[
        n_tot_vals_2d.min(), n_tot_vals_2d.max(),
        n_higgs_vals.min(), n_higgs_vals.max()
    ],
    aspect='auto'
)
cbar = plt.colorbar(im)
cbar.set_label("Midpoint replica statistical power Z")


plt.xlabel(r"$n_{\rm tot} = s + b$ (post-filter)")
plt.ylabel(r"$n_{\rm higgs} = s$ (post-filter)")
plt.title("Replica-derived midpoint power map")


# Overlay isocontours for reference thresholds
Z_levels = [2.0, 3.0, 5.0]
CS = plt.contour(
    N_tot_grid,
    S_grid,
    Z_med_grid,
    levels=Z_levels,
    colors='white',
    linewidths=1.0
)
plt.clabel(CS, inline=True, fontsize=8, fmt="%.0fσ")


plt.tight_layout()
plt.show()

# %%
# Employ reference anticipated yields post-selection
n_s_nom = s0
n_b_nom = b0


n_toys_hist = 20000
Z_toys_nom = run_toys(n_s_nom, n_b_nom, n_toys=n_toys_hist)


Z_med_toys = np.median(Z_toys_nom)
Z_as_nom = Z_asymptotic(n_s_nom, n_b_nom)


print(f"Analytical (theoretical) Z at reference yields: {Z_as_nom:.3f}")
print(f"Replica-based midpoint Z at reference yields:  {Z_med_toys:.3f}")


plt.figure(figsize=(7,5))
plt.hist(Z_toys_nom, bins=50, histtype='step', density=True)
plt.xlabel("Replica discovery statistical power Z")
plt.ylabel("Frequency density")
plt.title("Step 4: Z distribution from stochastic trials (reference yields)")


# Highlight medians
plt.axvline(Z_med_toys, color='C1', linestyle='-', label=f"Replica midpoint Z = {Z_med_toys:.2f}")
plt.axvline(Z_as_nom, color='C2', linestyle='--', label=f"Analytical Z = {Z_as_nom:.2f}")


plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# ============================================================
# LAB 8 — OBJECTIVE 4
# Authenticate power via stochastic count-based experiments
# ============================================================


from scipy.stats import poisson, norm


# ----‐---------------------------------------------------
# (1) "Employ acquired pass-rates from your analysis"
# ----‐---------------------------------------------------
# Sourced from optimized 3-factor filtering (Lab 7)
eff_higgs = best["sig_mask"].sum() / len(h)     # ε_s
eff_qcd   = best["bkg_mask"].sum() / len(q)     # ε_b


print("=== Acquisition Pass-Rates ===")
print(f"Signal pass-rate ε_s = {eff_higgs:.6f}")
print(f"Background pass-rate ε_b = {eff_qcd:.6f}")


# ----‐---------------------------------------------------
# (2) "Instantiate Poisson ensemble with mean = n_higgs and n_qcd"
# ----‐---------------------------------------------------
n_higgs = N_higgs_nom      # e.g. 100
n_qcd   = N_qcd_nom        # e.g. 20000


# Anticipated yields POST-filtering (for analytical comparison)
s_exp = n_higgs * eff_higgs
b_exp = n_qcd   * eff_qcd


print("\n=== Anticipated post-filter yields ===")
print(f"s = n_higgs * ε_s = {s_exp:.3f}")
print(f"b = n_qcd   * ε_b = {b_exp:.3f}")


# Analytical (eq.97) power (Lab 7)
Z_asimov_nom = float(Z_asymptotic(s_exp, b_exp))
print(f"Analytical Z (eq.97) = {Z_asimov_nom:.3f}")


# ----‐---------------------------------------------------
# (3) "Conduct numerous iterations"
# ----‐---------------------------------------------------
n_toys = 20000
rng = np.random.default_rng(12345)
Zs = []


# Under noise-only null:
b_mean_after = n_qcd * eff_qcd


# ----‐---------------------------------------------------
# (4) "Per iteration, sample Higgs & QCD production counts"
#      draw from Poisson(n_higgs), Poisson(n_qcd)
# (5) "Compute pass-through after filter"
#      implement Binomial thinning
# (6) "Combine post-filter totals"
# (7) "Derive exact power for null foundation"
# ----‐---------------------------------------------------
for _ in range(n_toys):


    # Pre-filter particle production
    Ns_prod = rng.poisson(n_higgs)
    Nb_prod = rng.poisson(n_qcd)


    # Post-filter particle survival
    Ns_sel = rng.binomial(Ns_prod, eff_higgs) if Ns_prod > 0 else 0
    Nb_sel = rng.binomial(Nb_prod, eff_qcd)   if Nb_prod > 0 else 0


    # Cumulative post-filter
    N_obs = Ns_sel + Nb_sel


    # Precise p under noise foundation:
    p = poisson.sf(N_obs - 1, mu=b_mean_after) if b_mean_after > 0 else 1.0


    # Map to Z metric
    Z = norm.isf(p) if p > 0 else np.inf
    Zs.append(Z)


Zs = np.array(Zs)
finite = np.isfinite(Zs)
Z_med_toys = np.median(Zs[finite])


print("\n=== Replica outcomes ===")
print(f"Replica midpoint Z = {Z_med_toys:.3f}")


# ----‐---------------------------------------------------
# (8) "Ultimately, render histogram of power for scenario
#      with n_higgs and n_qcd matched to reference yields,
#      highlighting midpoint from stochastic trials
#      and formula from step (1)"
# ----‐---------------------------------------------------


plt.figure(figsize=(7,5))
plt.hist(Zs[finite], bins=60, alpha=0.7)
plt.axvline(Z_med_toys,   color='k', linestyle='--', label=f"Replica midpoint = {Z_med_toys:.2f}")
plt.axvline(Z_asimov_nom, color='r', linestyle='--', label=f"Analytical Z = {Z_asimov_nom:.2f}")
plt.xlabel("Statistical Power Z")
plt.ylabel("Iteration count")
plt.title("Objective 4: Z distribution from stochastic count trials")
plt.legend()
plt.tight_layout()
plt.show()

# %%
toy_Z_map = np.zeros_like(Z_mesh)


n_toys_2D = 800
for i in range(S_mesh.shape[0]):
    for j in range(S_mesh.shape[1]):
        s = S_mesh[i,j]
        b = B_mesh[i,j]


        # invert (s,b) to generate pre-filter production expectations
        n_h = s / eff_higgs
        n_b = b / eff_qcd


        Ztrial = []
        for _ in range(n_toys_2D):
            Ns = rng.poisson(n_h)
            Nb = rng.poisson(n_b)


            Ns_sel = rng.binomial(Ns, eff_higgs) if Ns>0 else 0
            Nb_sel = rng.binomial(Nb, eff_qcd)   if Nb>0 else 0


            Nobs = Ns_sel + Nb_sel


            p = poisson.sf(Nobs-1, mu=b)
            Z = norm.isf(p) if p>0 else 0
            Ztrial.append(Z)


        toy_Z_map[i,j] = np.median(Ztrial)


plt.figure(figsize=(7,5))
plt.pcolormesh(S_mesh + B_mesh, S_mesh, toy_Z_map, shading="auto")
plt.colorbar(label="Replica Midpoint Z")
plt.xlabel("N_aggregate = S+B")
plt.ylabel("S")
plt.title("Objective 4 Validation: 2D Power Map (Replica Midpoint Z)")
plt.show()

# %%
# --- TASK 5: Import variable-intensity pseudo-data ---


pseudo_high = load_pseudo_data(f"data_highLumi_{SAMPLE}.h5")
pseudo_low  = load_pseudo_data(f"data_lowLumi_{SAMPLE}.h5")


print("High-intensity pseudo-data:", pseudo_high.shape)
print("Low-intensity pseudo-data :", pseudo_low.shape)


# --- TASK 5A: Unfiltered (baseline) comparison ---


def plot_raw_comparison(pseudo_df, title="Recorded vs Projected (Baseline)"):
    recorded = pseudo_df["mass"]


    # theoretical expectation = Higgs + QCD via sim
    exp_sig = h["mass"]
    exp_bkg = q["mass"]


    plt.figure(figsize=(10, 6))


    bins = 60


    # Stacked scaled histogram for sim with standardization
    plt.hist([exp_bkg, exp_sig], bins=bins,
             range=(0, 300),
             density=True,
             stacked=True,
             alpha=0.8,
             label=["Projected Noise (Sim)", "Projected Signal (Sim)"],
             color=['sandybrown', 'lightgreen'],
             edgecolor='none')


    # Recorded data as shaded histogram
    plt.hist(recorded, bins=bins,
             range=(0, 300),
             density=True,
             alpha=0.5,
             label="Recorded Pseudo-Data",
             color='steelblue',
             edgecolor='darkblue',
             linewidth=0.5)


    plt.xlabel("Invariant mass (GeV)", fontsize=12)
    plt.ylabel("Normalized frequency", fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_raw_comparison(pseudo_high, title="High-Intensity: Recorded vs Projected (Baseline)")
plot_raw_comparison(pseudo_low, title="Low-Intensity: Recorded vs Projected (Baseline)")


# --- TASK 5B: Execute cuts, then contrast ---

def apply_selection_df(df, mass_low=120, mass_high=130, d2_max=None, t21_max=None):
    """Apply optimized filtering to arbitrary dataframe"""
    if d2_max is None:
        d2_max = best['d2_max']
    if t21_max is None:
        t21_max = best['t21_max']

    sel = (
        (df["mass"] >= mass_low) & (df["mass"] <= mass_high) &
        (df["d2"] <= d2_max) &
        (df["t21"] <= t21_max)
    )
    return df[sel]


def plot_after_cuts(pseudo_df, title="Recorded vs Projected (Post Cuts)"):


    obs_sel = apply_selection_df(pseudo_df)["mass"]
    sig_sel = apply_selection_df(h)["mass"]
    bkg_sel = apply_selection_df(q)["mass"]


    # Rescale sim to match recorded population for meaningful juxtaposition
    N_obs = len(obs_sel)
    N_mc_sig = len(sig_sel) * signal_weight
    N_mc_bkg = len(bkg_sel) * background_weight
    N_mc_tot = N_mc_sig + N_mc_bkg


    scale = N_obs / N_mc_tot if N_mc_tot > 0 else 1.0


    plt.figure(figsize=(10, 6))


    bins = 50


    # Stacked weighted histogram for sim
    plt.hist([bkg_sel, sig_sel], bins=bins,
             weights=[np.full(len(bkg_sel), background_weight * scale),
                      np.full(len(sig_sel), signal_weight * scale)],
             stacked=True,
             alpha=0.8,
             label=["Projected Noise (Sim)", "Projected Signal (Sim)"],
             color=['sandybrown', 'lightgreen'],
             edgecolor='none')


    # Recorded data as shaded histogram
    plt.hist(obs_sel, bins=bins,
             alpha=0.5,
             label=f"Recorded Pseudo-Data (N={N_obs})",
             color='steelblue',
             edgecolor='darkblue',
             linewidth=0.5)


    plt.xlabel("Invariant mass (GeV)", fontsize=12)
    plt.ylabel("Particle count", fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_after_cuts(pseudo_high, title="High-Intensity: Recorded vs Projected (Post Cuts)")
plot_after_cuts(pseudo_low, title="Low-Intensity: Recorded vs Projected (Post Cuts)")

# %%
# --- 6A. Implement tuned criteria to pseudo-data ---
pseudo = pseudo_high      # alternatively pseudo_low


obs_sel = apply_selection_df(pseudo)
N_obs = len(obs_sel)


print(f"Recorded post-filter particle count (N_obs) = {N_obs}")


# projected noise post-filter (from sim)
b_expected = best["b"]     # precomputed earlier
s_expected = best["s"]     # projected signal post-filter


print(f"Projected noise post-filter (b) = {b_expected:.3f}")
print(f"Projected signal post-filter (s) = {s_expected:.3f}")


# --- 6B. Evaluate RECORDED POWER employing precise Poisson framework ---
# noise-only p-value
if b_expected > 0:
    p_val = poisson.sf(N_obs - 1, mu=b_expected)
    Z_obs = norm.isf(p_val) if p_val > 0 else np.inf
else:
    p_val = 1.0
    Z_obs = 0.0


print(f"\nRecorded statistical power Z_obs = {Z_obs:.3f} σ")



# --- 6C. When Z < 5σ, estimate 95% CL ceiling on signal yield ---
def upper_limit_CL95(N_obs, b, confidence=0.95):
    """
    Resolve for s where:
       P(N >= N_obs | mean = s + b) = 1 - confidence   (single-tailed)
    """
    target_tail = 1 - confidence
    # sweep across feasible signal production ranges
    s_scan = np.linspace(0, 200, 5000)   # sufficient granularity for current lab
    pvals = poisson.sf(N_obs - 1, mu=s_scan + b)
    idx = np.argmin(np.abs(pvals - target_tail))
    return s_scan[idx]


if Z_obs < 5:
    s95 = upper_limit_CL95(N_obs, b_expected)
    print(f"95% CL upper ceiling on signal yield: s < {s95:.2f}")
else:
    print("Recorded statistical power ≥ 5σ → Upper limit calculation omitted.")

# %% [markdown]
# ## Analysis of Power Formula Variants
# 
# 
# Leveraging the optimized three-factor filtering (mass, d2, and t21),  
# the projected signal and noise yields turned out as:
# 
# 
# S ≈ 14.04  
# B ≈ 0.40
# 
# 
# ---
# 
# 
# ### 1. Lab 5 Approach
# 
# 
# The elementary power computation reads:
# 
# 
#     Z_basic = S / sqrt(B)
# 
# 
# Computed:
# 
# 
#     Z_basic ≈ 22.19
# 
# 
# This turns artificially inflated when background B gets minuscule.  
# The formula expands infinitely as B → 0, rendering it inaccurate once the  
# filtering severely suppresses background.
# 
# 
# ---
# 
# 
# ### 2. Lab 7 (Cowan et al.) Exact Formula
# 
# 
# The superior central projected power relies on:
# 
# 
#     Z_exact = sqrt( 2 * ( (S+B)*ln(1 + S/B) - S ) )
# 
# 
# Computed:
# 
# 
#     Z_exact ≈ 8.69
# 
# 
# This formula respects sparse-count Poisson framework appropriately and avoids  
# explosion when B shrinks.
# 
# 
# ---
# 
# 
# ### 3. Conditions for Formula Concordance
# 
# 
# Both power calculations accord when:
# 
# 
# - S << B  
# - the experiment operates background-dominated  
# - Poisson samples approximate Gaussian distribution
# 
# 
# ---
# 
# 
# ### 4. Scenarios of Disagreement
# 
# 
# Formulas diverge substantially when:
# 
# 
# - B remains diminutive  
# - S/B rises substantially  
# - aggressive filtering drops background toward null  
# 
# 
# Precisely this configuration applies for our m

# %%
# ============================================================
# BONUS: ML Event Selection for Maximum Asymptotic Significance
# ============================================================

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("BONUS: ML-Based Event Selection Optimization")
print("=" * 60)

# ---- Step 1: Prepare training data ----
feature_cols = ['mass', 'd2', 't21', 'pt', 'eta', 'phi']
X = df_train[feature_cols].values
y = df_train['__label__'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ---- Step 2: Train Gradient Boosting Classifier ----
clf = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
)
clf.fit(X_train, y_train)

# Get ML scores
h['ml_score'] = clf.predict_proba(h[feature_cols])[:, 1]
q['ml_score'] = clf.predict_proba(q[feature_cols])[:, 1]

# ---- Step 3: Optimize ML threshold for asymptotic significance ----
score_thresholds = np.linspace(0.3, 0.99, 200)
Z_asym_vals, eff_sig_vals, eff_bkg_vals, s_vals, b_vals = [], [], [], [], []

for threshold in score_thresholds:
    sig_pass = h['ml_score'] >= threshold
    bkg_pass = q['ml_score'] >= threshold

    eff_sig = sig_pass.sum() / len(h)
    eff_bkg = bkg_pass.sum() / len(q)

    s = eff_sig * N_higgs_nom
    b = eff_bkg * N_qcd_nom

    Z_asym = float(Z_asymptotic(s, b)) if b > 0 else 0

    Z_asym_vals.append(Z_asym)
    eff_sig_vals.append(eff_sig)
    eff_bkg_vals.append(eff_bkg)
    s_vals.append(s)
    b_vals.append(b)

# Find optimal threshold
Z_asym_vals = np.array(Z_asym_vals)
best_idx = np.argmax(Z_asym_vals)
best_threshold = score_thresholds[best_idx]
best_Z_asym = Z_asym_vals[best_idx]
best_s = s_vals[best_idx]
best_b = b_vals[best_idx]
best_eff_sig = eff_sig_vals[best_idx]
best_eff_bkg = eff_bkg_vals[best_idx]

print(f"Optimal ML Score Threshold: {best_threshold:.3f}")
print(f"Signal efficiency: {best_eff_sig:.4f}")
print(f"Background efficiency: {best_eff_bkg:.6f}")
print(f"Expected S = {best_s:.2f}, B = {best_b:.2f}")
print(f"Expected Z_asymptotic = {best_Z_asym:.2f} σ")

# ---- Step 4: ROC curve ----
y_pred_test = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_test)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 4))

# Left: ROC curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR (Background Efficiency)')
plt.ylabel('TPR (Signal Efficiency)')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# Right: Significance vs threshold
plt.subplot(1, 2, 2)
plt.plot(score_thresholds, Z_asym_vals, 'b-', linewidth=2)
plt.axvline(best_threshold, color='r', linestyle='--',
            label=f'Optimal = {best_threshold:.3f}')
plt.xlabel('ML Score Threshold')
plt.ylabel('Z_asymptotic')
plt.title('Significance Optimization')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---- Step 5: Final comparison ----
print(f"\n{'='*50}")
print("COMPARISON WITH RECTANGULAR CUTS")
print(f"{'='*50}")
print(f"Rectangular cuts: Z = {best['Z_asym']:.2f} σ")
print(f"ML selection:    Z = {best_Z_asym:.2f} σ")
print(f"Improvement:     {best_Z_asym - best['Z_asym']:+.2f} σ")

# Feature importance
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': clf.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nFeature Importance:\n{importance_df.to_string(index=False)}")
