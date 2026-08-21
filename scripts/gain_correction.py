"""The replicate gain correction: one exponent per distorted replicate, clamped.

A replicate sometimes resolves less of an assay's range than its two siblings do:
KSR1 C-term activity rep 3 spans half the range of reps 1 and 2, so a variant that
reps 1 and 2 put at 8x wild type it puts at 2.8x. That is a property of the
measurement rather than of the analysis -- an independent scoring of the same
reads recovers the same gains -- and it survives
replicate averaging, pulling the cell's mean toward the compressed replicate.

THE CORRECTION is one parameter:

    score' = score * 10 ** (clip(log10 score, lo, hi) * (a_ref / a_j - 1))

which away from the clamp is just `score ** (a_ref / a_j)`. In log-log space it is a
horizontal dilation of that replicate's axis about score = 1, so wild type -- and the
whole line through it -- cannot move, which the WT-relative scale and the DN
threshold both depend on.

`a_j` is the replicate's GAIN, identified from the off-diagonal covariances

    cov(i, j) = a_i a_j S

which involve no replicate's own variance and so are immune to that replicate's
noise. `a_ref` is the mean gain of the OTHER TWO replicates -- both of them, always,
even when one is itself flagged: averaging two siblings is what makes the target
worth aiming at, and comparing against a single sibling halves the averaging and
lets that one replicate's distortion define the reference.

This is deliberately NOT quantile matching. Matching a replicate's score
distribution to its siblings' equalises TOTAL variance, signal plus noise, so a
noisier replicate has its signal scaled wrongly. Measured: quantile matching left
KSR1 C-term rep 3 with a residual of 0.181 against 0.057 here, and it compressed
MEK1 abundance rep 1 (SD 0.126 -> 0.069) because that replicate's extra spread was
mostly noise rather than range.

FLAGGING -- a replicate is corrected only if its gain differs from the cell's median
gain by at least `MAX_GAIN_DEV` in log2, and only if the correction then reduces the
binned deviation from y = x by at least `MIN_IMPROVEMENT`. Over the 174 evaluable
replicates the gain deviation has a median of 0.011, a 95th percentile of 0.204, a
99th of 0.496 and a maximum of 0.978, so 0.5 sits just under the 99th percentile and
selects the two genuine dynamic-range failures: KSR1 C-term activity rep 3 (0.978)
and MRAS activity rep 2 (0.810). Everything else comes through bit-identical.

THE CLAMP holds the log-offset constant outside the 1st and 99th percentiles of the
replicate's own fitting set. Without it the exponent extrapolates: 2.0 took a
standard from 32x wild type to 1050x, against ~78x in the sibling replicates. Beyond
the clamp the map has slope 1, so ranks and relative spacing are preserved and every
variant out there receives one multiplicative factor -- the honest statement that
past what was measured no further rescaling is claimed.

NO SHAPE STEP. A second step used to follow the gain -- a monotone PCHIP spline
contributing only its bend -- and it was removed on 2026-08-19 because it was
measured and it does not help:

  * On held-out variants (fitted on half the missense, scored on the other half) it
    LOSES to the plain exponent: KSR1 C-term rep 3 scores 0.197 gain-only against
    0.228 spline-only and 0.248 for both; MRAS rep 2 scores 0.165 against 0.170 and
    0.175. At 4 knots the map was not even monotone.
  * The bend it would fit is not real curve shape. It reproduces across random
    halves at r = 0.99, but that proves nothing, because regression dilution is a
    systematic bias rather than sampling noise. Simulating three replicates exactly
    linear in log space, with each cell's measured gains and noise, reproduces a
    bend of the same sign and shape (r = 0.82 / 0.95) at comparable or larger
    magnitude. Fitting it pulls the replicate toward its siblings' noise: better
    agreement, not a better measurement.
  * `binned_dev` bins on the consensus, so it is minimised by a map that
    over-shrinks the replicate toward its siblings. A flexible spline can shrink
    more than one exponent can, so the acceptance metric was biased in the spline's
    favour -- and the spline lost anyway.

Curvature is still computed and logged (`curvature`) as a diagnostic, but nothing is
flagged on it.

THE STANDARDS RIDE THE SAME MAP, and so does the empty-vector control, because
numerator and calibration have to stay consistent. That is the lesser of two
imperfect options, and the size of both is worth knowing.

The standards do not lie on the variants' curve -- at a replicate value where the
variants put the consensus at 0.68-1.21 the standards put it at 1.31-1.55, the same
direction in KSR1 C-term, RET and ARAF C-term -- so they cannot anchor the fit, and
the clamped map does not land them where their own consensus wants:

  MRAS rep 2      clamp applies x1.27, the standards want x1.38 (slight under-shoot)
                  calibration slope 0.411 -> 0.523 against siblings 0.554 / 0.577,
                  so the cell's slope spread falls from 1.40 to 1.10
  KSR1 C-term
  rep 3           clamp applies x3.55, the standards want x2.36 (1.5x OVER-shoot)
                  calibration slope 4.231 -> 15.011 against siblings 9.864 / 10.120,
                  so the slope flips from too low to too high, though the spread
                  still falls from 2.39 to 1.52

Leaving the standards uncorrected is worse, not better: rep 3's variants would be
scaled x2.00 while its slope stayed at 4.231, putting its standard-adjusted scores
out by a factor of 2.00 x 9.864 / 4.231 = 4.7 relative to its siblings, against the
1.5 the overshoot costs. So the standards are corrected, and the residual is
recorded here rather than hidden.

REFUSALS are recorded rather than silent: `MIN_SNR` (in a noise-dominated cell the
gain rests on small off-diagonal covariances -- below SNR 1 only 27% of replicate
pairs improve, against 98% above 4), a non-positive off-diagonal covariance, all
three replicates flagged so no sound reference remains, fewer than `MIN_VARIANTS`
usable fitting variants, and a fitting range whose middle 95% excludes wild type
(which would make the clamped map move wild type).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Flag threshold on |log2 gain / cell median gain|. See the module docstring for
#: the distribution this is drawn from; 0.5 is a factor of 1.41.
MAX_GAIN_DEV = 0.5
#: Below this signal-to-noise ratio the gain estimate is not trustworthy.
MIN_SNR = 1.0
#: Minimum usable fitting variants in a cell.
MIN_VARIANTS = 500
#: Bins for `binned_dev`.
DEV_BINS = 12
#: log2 per decade. The metrics are computed on log10 arrays and converted on the
#: way out so that everything reported is in log2, like `MAX_GAIN_DEV`.
LOG2_10 = float(np.log2(10))
#: A correction must buy at least this much reduction in `binned_dev` to be applied.
#: Without it, replicates were being rewritten for nothing: MEK2 abundance rep 3
#: moved its scores by 3.8% to take the deviation from 0.1649 to 0.1634, and GRB2
#: activity rep 1 by 4.5% for 0.0007. Changing data has to earn it.
MIN_IMPROVEMENT = 0.01 * LOG2_10
#: Percentiles of the fitting set that bound the clamp.
CLAMP_PCT = (1.0, 99.0)


def binned_dev(x: np.ndarray, y: np.ndarray, nbin: int = DEV_BINS) -> float:
    """Largest |median(y) - median(x)| over quantile bins of x, in LOG2 units.

    The acceptance criterion: one number covering shift, tilt and bend together,
    which is what "sits on the diagonal" means. Preferred to a Deming tilt, which
    is unstable on the bimodal cells -- it read 2.3 on a cloud that had visibly
    improved, because the principal axis flips between modes.

    Note that binning on x = the consensus means even a perfect correction scores
    above zero: conditioning on a noisy consensus selects variants whose consensus
    is extreme, which are biased outward relative to the truth. The metric is
    therefore minimised by a map that over-shrinks toward the siblings, which is why
    it is used only as an accept/reject gate on a map fitted independently of it.
    """
    e = np.quantile(x, np.linspace(0, 1, nbin + 1))
    e[0] -= 1e-9
    e[-1] += 1e-9
    idx = np.digitize(x, e) - 1
    out = [abs(np.median(y[idx == b]) - np.median(x[idx == b]))
           for b in range(nbin) if (idx == b).sum() >= 20]
    return float(np.max(out)) * LOG2_10 if len(out) >= 6 else np.nan


def curvature(x: np.ndarray, y: np.ndarray, nbin: int = 20) -> float:
    """Deviation of the binned relation from its own best-fit line, in LOG2 units.

    Diagnostic only -- nothing is flagged on it. Distinct from `binned_dev` in that
    it removes the slope, so a pure gain problem reads near zero and a bend at the
    right slope reads high.
    """
    e = np.quantile(x, np.linspace(0, 1, nbin + 1))
    e[0] -= 1e-9
    e[-1] += 1e-9
    idx = np.digitize(x, e) - 1
    bx, by = [], []
    for b in range(nbin):
        m = idx == b
        if m.sum() >= 20:
            bx.append(np.median(x[m]))
            by.append(np.median(y[m]))
    if len(bx) < 6:
        return np.nan
    bx, by = np.array(bx), np.array(by)
    sl, ic = np.polyfit(bx, by, 1)
    return float(np.max(np.abs(by - (sl * bx + ic)))) * LOG2_10


def gains_and_snr(L: np.ndarray):
    """Per-replicate gain (median 1) and the cell's signal-to-noise ratio.

    `L` is (n, 3) log10 scores. The gain is identified only up to a global factor;
    dividing by the MEDIAN rather than the geometric mean is what leaves sound
    replicates where they are instead of rescaling all three to meet the outlier.
    """
    C = np.cov(L, rowvar=False)
    c12, c13, c23 = C[0, 1], C[0, 2], C[1, 2]
    S = float(np.mean([c12, c13, c23]))
    N = float(np.mean([C[i, i] for i in range(3)]) - S)
    snr = float(np.sqrt(S / N)) if (S > 0 and N > 0) else np.nan
    if min(c12, c13, c23) <= 0:
        return None, snr
    a = np.array([np.sqrt(c12 * c13 / c23),
                  np.sqrt(c12 * c23 / c13),
                  np.sqrt(c13 * c23 / c12)])
    return a / np.median(a), snr


def gain_map(ratio: float, lo: float, hi: float):
    """The correction as a plain function of score, for reuse on other populations.

    Returns a callable mapping raw scores to corrected scores, so the standards and
    the empty-vector control barcodes can be put through the same map as the
    variants rather than left on a different scale. Non-positive and missing scores
    pass through untouched.

    `lo`/`hi` are the clamp bounds in log10. Written as an offset clipped at the
    bounds rather than as a piecewise formula, which makes it monotone by
    construction and exact at the boundary.
    """
    def apply(score):
        s = np.asarray(score, dtype=float)
        out = s.copy()
        ok = np.isfinite(s) & (s > 0)
        if ok.any():
            y = np.log10(s[ok])
            out[ok] = 10.0 ** (y + np.clip(y, lo, hi) * (ratio - 1.0))
        return out
    return apply


def correct_cell(scores: pd.DataFrame, cols, fit_mask, cell: dict | None = None):
    """Correct the distorted replicates of ONE cell (library x assay x treatment).

    Parameters
    ----------
    scores    per-variant frame for the cell, carrying `cols`
    cols      the three raw replicate score columns, in order
    fit_mask  boolean over `scores`: the variants the gain is fitted on. Missense
              passing the barcode-count cutoff -- the correction is estimated on the
              population it is best measured in and then applied to every class.
    cell      identifying fields copied into the log records

    Returns
    -------
    corrected  (n, 3) array aligned to `scores`, equal to the raw values wherever no
               correction applies
    records    one dict per replicate, always three, carrying the gain, its
               deviation, the curvature, the SNR, and either the correction or the
               reason there is none
    maps       list of three callables-or-None; `maps[j]` puts any other population
               of raw scores from this cell's replicate j through the same map
    """
    base = dict(cell or {})
    V = scores[list(cols)].apply(pd.to_numeric, errors="coerce")
    raw = V.to_numpy(dtype=float)
    corrected = raw.copy()
    maps: list = [None, None, None]

    good = np.isfinite(raw).all(axis=1) & (raw > 0).all(axis=1)
    fit = np.asarray(fit_mask, dtype=bool) & good
    if fit.sum() < MIN_VARIANTS:
        return corrected, [{**base, "rep": j + 1, "n_fit": int(fit.sum()),
                            "status": "too few fitting variants"}
                           for j in range(3)], maps

    Lf = np.log10(raw[fit])
    a, snr = gains_and_snr(Lf)
    if a is None or not np.isfinite(snr) or snr < MIN_SNR:
        why = ("non-positive covariance" if a is None
               else f"refused: SNR {snr:.2f} < {MIN_SNR}")
        return corrected, [{**base, "rep": j + 1, "snr": snr,
                            "n_fit": int(fit.sum()), "status": why}
                           for j in range(3)], maps

    gd = np.abs(np.log2(a))
    cv = np.array([curvature(Lf[:, [k for k in range(3) if k != j]].mean(axis=1),
                             Lf[:, j]) for j in range(3)])
    flagged = gd >= MAX_GAIN_DEV
    common = dict(base, snr=snr, n_fit=int(fit.sum()))
    if flagged.all():
        return corrected, [{**common, "rep": j + 1, "gain": a[j],
                            "gain_dev_log2": gd[j], "curvature": cv[j],
                            "status": "all three flagged; no sound reference"}
                           for j in range(3)], maps

    Lg = np.log10(raw[good])
    fit_of_good = fit[good]
    records = []
    for j in range(3):
        others = [k for k in range(3) if k != j]
        rec = {**common, "rep": j + 1, "gain": a[j], "gain_dev_log2": gd[j],
               "curvature": cv[j],
               "reference_reps": "+".join(str(k + 1) for k in others)}
        if not flagged[j]:
            records.append({**rec, "status": "unchanged", "shift_log2": 0.0})
            continue

        cons = Lf[:, others].mean(axis=1)
        dev = binned_dev(cons, Lg[fit_of_good, j])
        ratio = float(np.mean(a[others]) / a[j])
        lo, hi = np.percentile(Lf[:, j], CLAMP_PCT)
        rec.update(dev_before=dev, clamp_lo=float(10 ** lo),
                   clamp_hi=float(10 ** hi), gain_ratio=ratio)
        if not (lo <= 0.0 <= hi):
            # the clamped map would move wild type, which nothing downstream can
            # tolerate: the scores are WT-relative and the DN threshold sits at 1
            records.append({**rec, "shift_log2": 0.0,
                            "status": "refused: middle 95% excludes wild type"})
            continue

        f = gain_map(ratio, lo, hi)
        cand = f(raw[good, j])
        d_new = binned_dev(cons, np.log10(cand[fit_of_good]))
        if not (np.isfinite(d_new) and d_new < dev - MIN_IMPROVEMENT):
            records.append({**rec, "dev_after": d_new, "shift_log2": 0.0,
                            "status": "no improvement; unchanged"})
            continue

        corrected[good, j] = cand
        maps[j] = f
        records.append({
            **rec, "dev_after": d_new,
            "flagged_for": f"dynamic range {gd[j]:.2f}",
            # over the FITTING population, not every affected row: the fitting
            # set is a stable, well-measured population, so the number is
            # comparable between runs and between cells. Rows that later fail the
            # barcode cutoff are still corrected -- they are simply not what this
            # diagnostic is measured on.
            "shift_log2": float(np.median(np.abs(
                np.log2(cand[fit_of_good] / raw[good, j][fit_of_good])))),
            "status": f"corrected: gain x{ratio:.3f}"})
    return corrected, records, maps
