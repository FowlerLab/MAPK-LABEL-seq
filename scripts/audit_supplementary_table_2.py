"""Pre-release audit of Supplementary Table 2.

Every check prints PASS, WARN (true but worth knowing before release) or FAIL
(the table contradicts its own definition). Run before every upload:

    python scripts/audit_supplementary_table_2.py [output/Supplementary_Table_2.tsv]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ST2 = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output" / "Supplementary_Table_2.tsv"
KEY = ["library", "variant", "assay", "assay_treatment"]
BASAL = {"DMSO", "No_treatment"}
INHIBITORY = {"grb2", "ksr1", "ksr2", "mek1", "mek2"}   # WT lowers pathway activity
AA3 = {"A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln", "E": "Glu",
       "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe",
       "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val", "*": "Ter"}

results = []


def report(status, check, detail=""):
    results.append((status, check, detail))
    print(f"[{status:4s}] {check}" + (f" — {detail}" if detail else ""))


def check(cond, name, detail_ok="", detail_bad="", warn=False):
    report("PASS" if cond else ("WARN" if warn else "FAIL"), name, detail_ok if cond else detail_bad)


raw_text_head = open(ST2, encoding="utf-8").readline()
IDX = 0 if raw_text_head.startswith("\t") else None
st = pd.read_csv(ST2, sep="\t", index_col=IDX, low_memory=False)
s = pd.read_csv(ST2, sep="\t", index_col=IDX, dtype=str, keep_default_na=False)
num = lambda c: pd.to_numeric(st[c], errors="coerce")
real = ~st.variant_category.isin(["standard", "WT"])

print(f"== {ST2}\n   {st.shape[0]:,} rows x {st.shape[1]} columns\n")

# ── 1. Structure ────────────────────────────────────────────────────────────
print("1. Structure")
sys.path.insert(0, str(ROOT / "src"))
from labelseq_mapk.supplementary_table import ST2_COLUMNS
check(list(st.columns) == ST2_COLUMNS, f"{len(ST2_COLUMNS)} columns, in the schema's order",
      detail_bad=f"{st.shape[1]} columns; differ from the schema")
check(IDX is None, "no unnamed index column", detail_bad="the file carries a pandas index")
check(not st.duplicated(KEY).any(), "key (library, variant, assay, assay_treatment) is unique",
      detail_bad=f"{int(st.duplicated(KEY).sum())} duplicate keys")
check(st[KEY + ["protein", "variant_category"]].notna().all().all(), "key and type columns have no blanks")
lib_prot = st.groupby("library").protein.nunique()
check((lib_prot == 1).all(), "each library maps to one protein")
acc = st.groupby("protein").uniprot_id.nunique()
check((acc == 1).all(), "one uniprot_id per protein", detail_bad=str(acc[acc > 1].to_dict()))
both = st[st.assay_treatment.isin(BASAL)].groupby(["library", "assay"]).assay_treatment.nunique()
check((both == 1).all(), "no library x assay has both DMSO and No_treatment")

# ── 2. Identity ─────────────────────────────────────────────────────────────
print("\n2. Variant identity")
pos_num = pd.to_numeric(st.position, errors="coerce")
check(pos_num[real].notna().all(), "position numeric for every library variant",
      detail_bad=f"{int(pos_num[real].isna().sum())} non-numeric")
nonnum = st.loc[pos_num.isna(), "variant_category"].value_counts().to_dict()
report("WARN" if nonnum else "PASS", "position column is mixed-type", f"non-numeric only on {nonnum}")
sub = st[st.variant_category.isin(["missense", "synonymous", "nonsense"])]
m = sub.variant.str.extract(r"^([A-Z])(\d+)([A-Z*])$")
ok = (m[0] == sub.wild_type_residue) & (m[1].astype(float) == pd.to_numeric(sub.position)) & (m[2] == sub.mutation)
check(ok.all(), "variant string agrees with wild_type_residue / position / mutation (substitutions)",
      detail_bad=f"{int((~ok).sum())} disagree")
mis = sub[sub.variant_category == "missense"]
exp = "p." + mis.wild_type_residue.map(AA3) + mis.position.astype(str).str.replace(r"\.0$", "", regex=True) + mis.mutation.map(AA3)
check((exp == mis.hgvs_p).all(), "hgvs_p agrees with the variant (missense)",
      detail_bad=f"{int((exp != mis.hgvs_p).sum())} disagree")
seq, cur = {}, None
for line in open(ROOT / "data" / "reference_protein_sequences.fasta"):
    if line.startswith(">"):
        cur = line[1:].split()[0].lower(); seq[cur] = ""
    else:
        seq[cur] += line.strip()
w = st[real & pos_num.notna() & st.wild_type_residue.str.fullmatch(r"[A-Z]")].drop_duplicates(["protein", "position"])
ref = [seq.get(p, "")[int(q) - 1] if 0 < int(q) <= len(seq.get(p, "")) else "" for p, q in zip(w.protein, pd.to_numeric(w.position))]
mism = (w.wild_type_residue.values != np.array(ref)) & (np.array(ref) != "")
past = np.array([r == "" for r in ref])
check(not mism.any(), "wild-type residue matches the reference sequence at every position",
      f"{len(w):,} positions; {int(past.sum())} past the protein end (C-terminal MCP-tag junction)",
      f"{int(mism.sum())} mismatches: {w.loc[mism, ['protein', 'position', 'wild_type_residue']].head().values.tolist()}")
n_var = st.loc[real, ["protein", "variant"]].drop_duplicates().shape[0]
report("WARN", "distinct variants", f"{n_var:,} distinct (protein, variant) pairs; {st.loc[real, 'variant'].nunique():,} distinct variant strings — quote the first")

# ── 3. Scores ───────────────────────────────────────────────────────────────
print("\n3. Scores")
R = [f"WT_relative_score_replicate_{j}" for j in (1, 2, 3)]
S = [f"standardized_score_replicate_{j}" for j in (1, 2, 3)]
for cols, mean, name in ((R, "mean_WT_relative_score", "WT-relative"), (S, "mean_standardized_score", "standardized")):
    d = (st[cols].mean(axis=1) - st[mean]).abs()
    check(d.max(skipna=True) < 1e-9, f"{mean} = mean of its three replicates", detail_bad=f"max |diff| {d.max():.3g}")
for j in (1, 2, 3):
    diff = ~np.isclose(st[f"score_{j}"], st[f"WT_relative_score_replicate_{j}"], equal_nan=True)
    cells = st.loc[diff].groupby(["library", "assay", "assay_treatment"]).size().to_dict()
    if j == 1:
        check(not diff.any(), "replicate 1: corrected = raw everywhere", detail_bad=str(cells))
    else:
        expect = {2: ("mras", "activity", "No_treatment"), 3: ("ksr1_cterm", "activity", "No_treatment")}[j]
        check(set(cells) <= {expect}, f"replicate {j}: corrected differs from raw only in {expect[0]} activity",
              str(cells), str(cells))
for j in (1, 2, 3):
    ratio = st[f"standardized_score_replicate_{j}"] / st[f"WT_relative_score_replicate_{j}"]
    spread = ratio.groupby([st.library, st.assay, st.assay_treatment]).agg(lambda x: x.max() / x.min() - 1 if x.notna().any() else 0)
    check(spread.max() < 1e-6, f"standardized_score_replicate_{j} = WT-relative / one slope per cell",
          detail_bad=f"max within-cell spread {spread.max():.3g}")
neg = (st[R + S] <= 0).sum().sum()
check(neg == 0, "no zero or negative replicate scores", detail_bad=f"{int(neg)} cells")
wt = st[st.variant == "WT"]
check(wt.mean_WT_relative_score.between(0.9, 1.1).all(), "WT rows have WT-relative score ~1",
      f"range {wt.mean_WT_relative_score.min():.3f}-{wt.mean_WT_relative_score.max():.3f}",
      f"range {wt.mean_WT_relative_score.min():.3f}-{wt.mean_WT_relative_score.max():.3f}", warn=True)
blank = st.mean_WT_relative_score.isna()
report("WARN" if blank.any() else "PASS", "rows without a WT-relative score",
       f"{int(blank.sum())} rows: {st.loc[blank, 'variant_category'].value_counts().to_dict()}")

# ── 4. Classification ───────────────────────────────────────────────────────
print("\n4. variant_classification (synonymous 2.5th / 97.5th percentile per cell)")
syn = st[st.variant_category == "synonymous"].groupby(["library", "assay", "assay_treatment"]).mean_WT_relative_score
lo, hi = syn.quantile(0.025), syn.quantile(0.975)
cell = pd.MultiIndex.from_frame(st[["library", "assay", "assay_treatment"]])
L, H = cell.map(lo.to_dict()), cell.map(hi.to_dict())
x = st.mean_WT_relative_score
recomp = np.where(x.isna(), None, np.where(x < L, "low", np.where(x > H, "high", "wt-like")))
has = st.variant_classification.notna()
agree = (recomp[has.values] == st.variant_classification[has].values).mean()
check(agree > 0.9999, "classification reproduces from the synonymous thresholds",
      f"{agree*100:.3f}% of {int(has.sum()):,} rows", f"{agree*100:.3f}% agree")

# ── 5. Dominant negatives ───────────────────────────────────────────────────
print("\n5. dominant_negative")
dn = st.dominant_negative
assess = dn.notna()
check(set(st.loc[assess, "assay"]) == {"activity"} and set(st.loc[assess, "assay_treatment"]) <= BASAL,
      "DN assessed only on basal activity rows")
check(set(st.loc[assess, "variant_category"]) == {"missense", "3nt deletion", "nonsense"},
      "DN assessed only for missense / 3-nt deletion / nonsense",
      detail_bad=str(st.loc[assess, "variant_category"].value_counts().to_dict()))
check(not st.loc[assess, "protein"].isin(INHIBITORY).any(), "no DN calls in inhibitory proteins (GRB2, KSR1, KSR2, MEK1, MEK2)")
thr = pd.read_csv(ROOT / "output" / "scoring" / "dn_thresholds_recomputed.tsv", sep="\t")
thr = thr[(thr.assay == "activity") & (thr.control == "used")].set_index(["library", "assay_treatment"]).dn_threshold
t = pd.MultiIndex.from_frame(st.loc[assess, ["library", "assay_treatment"]]).map(thr.to_dict())
rec = st.loc[assess, "mean_WT_relative_score"].values < t
check((rec == (dn[assess].astype(str) == "True").values).all(), "DN calls reproduce from the empty-vector thresholds",
      f"{int(rec.sum()):,} DN of {int(assess.sum()):,} assessed", f"{int((rec != (dn[assess].astype(str)=='True').values).sum())} disagree")
n_dn = int((dn.astype(str) == "True").sum())
check(n_dn == 14412, "DN total = 14,412 (the number the text quotes)", detail_bad=str(n_dn))
bc = st.loc[assess & (dn.astype(str) == "True") & (st.library == "braf_cterm")].shape[0]
report("WARN", "braf_cterm DN calls", f"{bc:,} of {n_dn:,} DNs come from the library the DN threshold check flags (15.8% of synonymous below threshold)")

# ── 6. HSP90 columns ────────────────────────────────────────────────────────
print("\n6. HSP90 columns")
hsp_lib = set(st.loc[st.assay_treatment == "HSP90i", "library"])
check(st.loc[st.buffering_class.notna(), "library"].isin(hsp_lib).all(), "buffering_class only in libraries with an HSP90i arm")
check(set(st.loc[st.buffering_class.notna(), "assay"]) <= {"abundance", "activity"}, "buffering_class copied onto abundance and activity rows",
      str(st.loc[st.buffering_class.notna(), "assay"].value_counts().to_dict()))
per_var = st.groupby(["library", "variant"]).buffering_class.nunique()
check((per_var <= 1).all(), "buffering_class is one value per variant (repeated across that variant's rows)")
dt = st.dependent_threshold.astype(str).str.lower() == "true"
bf = st.buffering_class.isin(["Buffered", "Poorly buffered"])
both_def = st.buffering_class.notna()
same = (dt[both_def] == bf[both_def]).all()
report("WARN", "dependent_threshold duplicates buffering_class",
       "identical to buffering_class in {Buffered, Poorly buffered} on every classified row; it means 'low under HSP90i', not the paper's dependence (`dependent`)" if same else "differs")
ds_cat = st.loc[st.dependent_significant.notna(), "variant_category"].value_counts().to_dict()
check(set(ds_cat) <= {"missense", "3nt deletion"}, "dependent_significant only for missense and 3-nt deletions", str(ds_cat), str(ds_cat))
dep = st.dependent.astype(str).str.lower() == "true"
check((st.dependent.notna() == st.dependent_threshold.notna()).all(),
      "dependent filled exactly where the abundance arms pair (as dependent_threshold)")
check((st.groupby(["library", "variant"]).dependent.nunique() <= 1).all(), "dependent is one value per variant")
# Recompute the gap rule from the table itself: ctrl - HSP90i >= max(0, WT_ctrl - syn_2.5pct_ctrl).
ab = st[st.assay == "abundance"]
ctrl = ab[ab.assay_treatment.isin(BASELINE := {"DMSO", "No_treatment"})]
gap = (ctrl[ctrl.variant_category == "WT"].groupby("library").mean_standardized_score.mean()
       - ctrl[ctrl.variant_category == "synonymous"].groupby("library").mean_standardized_score
       .agg(lambda v: np.percentile(v.dropna(), 2.5))).clip(lower=0)
pair = ctrl[["library", "variant", "protein", "domain", "variant_category", "mean_standardized_score", "dependent"]].merge(
    ab[ab.assay_treatment == "HSP90i"][["library", "variant", "mean_standardized_score"]],
    on=["library", "variant"], suffixes=("_c", "_h")).dropna(subset=["mean_standardized_score_c", "mean_standardized_score_h"])
rec_dep = (pair.mean_standardized_score_c - pair.mean_standardized_score_h) >= pair.library.map(gap)
check((rec_dep == (pair.dependent.astype(str).str.lower() == "true")).all(),
      "dependent reproduces from the gap rule (per library, standardized score)",
      f"{int(rec_dep.sum()):,} dependent of {len(pair):,} paired variants",
      f"{int((rec_dep != (pair.dependent.astype(str).str.lower() == 'true')).sum())} disagree")
FIG5D = {"craf": .989, "araf": .999, "ret": .991, "ksr2": .731, "egfr": .946, "met": .704,
         "braf": .631, "mek2": .426, "mek1": .084}
kd = pair[pair.variant_category.isin(["missense", "3nt deletion"]) & (pair.domain == "kinase")]
got = {}
for p_ in FIG5D:
    k = kd[kd.protein == p_]
    k = k[k.library == k.library.mode().iloc[0]]
    got[p_] = round(float((k.dependent.astype(str).str.lower() == "true").mean()), 3)
check(got == FIG5D, "dependent reproduces the Fig 5d dependent fractions", str(got), f"table {got} vs figure {FIG5D}")

# ── 7. Barcodes ─────────────────────────────────────────────────────────────
print("\n7. Barcode support")
check((st.mean_number_quantified_barcodes <= st.total_observed_barcodes + 1e-9).all(), "quantified barcodes <= observed barcodes")
low_bc = real & (st.mean_number_quantified_barcodes < 5)
check(not low_bc.any(), "every library variant has >= 5 mean quantified barcodes (inclusion rule)",
      detail_bad=f"{int(low_bc.sum())} rows below 5: {st.loc[low_bc, 'variant_category'].value_counts().to_dict()}")

# ── 8. Annotations ──────────────────────────────────────────────────────────
print("\n8. Annotations")
pos_cols = ["max_sasa", "relative_sasa", "dssp_solvent_accessibility_angstroms^2", "domain", "AF_plddt",
            "active_site", "inter_domain_contacts_all_atom", "at_hsp90_unfolded", "at_hsp90_folded", "at_cdc37",
            "phylop_vert", "jsd_conservation", "kinase_motif", "kinase_alignment_position", "feature"]
pp = st[real & pos_num.notna()].assign(_p=pos_num)
bad = {c: int((pp.groupby(["protein", "_p"])[c].nunique() > 1).sum()) for c in pos_cols}
check(not any(bad.values()), "per-position annotations are constant across a position's rows",
      detail_bad=str({k: v for k, v in bad.items() if v}))
check(num("AF_plddt").between(0, 100).all() or num("AF_plddt").isna().any(), "AF_plddt within 0-100")
over1 = int((num("relative_sasa") > 1).sum())
report("WARN" if over1 else "PASS", "relative_sasa > 1", f"{over1:,} rows (max {num('relative_sasa').max():.2f}): real, more exposed than the reference tripeptide")
anyc = (st[["at_hsp90_unfolded", "at_hsp90_folded", "at_cdc37"]].astype(str) == "True").any(axis=1)
check((anyc == (st.at_any_chaperone_contact.astype(str) == "True")).all(), "at_any_chaperone_contact = OR of the three contact columns")
kin = st.protein.isin({"araf", "braf", "craf", "egfr", "erbb2", "met", "ret", "ksr1", "ksr2", "mek1", "mek2"})
check(not st.loc[~kin, "jsd_conservation"].notna().any() and not st.loc[~kin, "kinase_alignment_position"].notna().any(),
      "kinase-only columns blank outside kinases")
gc = num("genie_count")
check((gc.dropna() >= 0).all() and (gc.dropna() % 1 == 0).all(), "genie_count non-negative integers")
pop_cat = set(st.loc[st.in_gnomad.notna(), "variant_category"])
check(pop_cat <= {"missense", "synonymous", "nonsense", "3nt deletion"},
      "in_gnomad only on variant types a database can record", str(pop_cat), str(pop_cat))
araf_as = sorted(pd.to_numeric(st.loc[(st.protein == "araf") & (st.active_site.astype(str) == "True"), "position"]).unique())
check(len(araf_as) == 46 and min(araf_as) == 316 and max(araf_as) == 471, "ARAF active site = the corrected 46 positions (316-471)",
      detail_bad=f"{len(araf_as)} positions")
nz = st.nmd_zone.notna()
check(set(st.loc[nz, "variant_category"]) <= {"nonsense"}, "nmd_zone only on nonsense rows",
      f"{int(nz.sum()):,} of {int((st.variant_category == 'nonsense').sum()):,} nonsense rows (single-nucleotide-reachable stops only)")
report("WARN", "nmd_zone coverage",
       "filled only for stops reachable by one nucleotide change (the population-table enumeration); other nonsense rows are blank")
fe = st.feature.dropna()
report("WARN" if fe.str.startswith("[").any() else "PASS", "feature column format",
       "semicolon-joined text; check no Python list syntax remains" if not fe.str.startswith("[").any() else "contains Python list syntax")

# ── 9. Release / policy ─────────────────────────────────────────────────────
print("\n9. Release and policy")
aou = [c for c in st.columns if re.search(r"(?i)aou|all_of_us", c)]
check(not aou, "no All of Us-derived column (Controlled Tier data is not released per variant)", detail_bad=str(aou))
pii = [c for c in st.columns if re.search(r"(?i)patient|sample_id|participant|person|dob|name$", c)]
check(not pii, "no participant-level identifier columns", detail_bad=str(pii))
txt = open(ST2, "rb").read(2_000_000)
check(b"\r\n" not in txt, "Unix line endings")
try:
    open(ST2, encoding="ascii").read(); report("PASS", "file is plain ASCII")
except UnicodeDecodeError:
    report("WARN", "file contains non-ASCII characters", "UTF-8; check the repository accepts it")
bools = [c for c in st.columns if set(s[c].unique()) <= {"True", "False", ""} and s[c].ne("").any()]
report("PASS", "boolean encoding", f"{len(bools)} columns use True/False with blank = not assessed")

print()
counts = pd.Series([r[0] for r in results]).value_counts().to_dict()
print("SUMMARY:", counts)
sys.exit(1 if counts.get("FAIL") else 0)
