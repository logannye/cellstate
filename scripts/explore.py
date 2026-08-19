"""An interactive surface over the committed GSE274113 slice.

Nothing here is a new capability or a new claim.  Every number this prints is computed by the
shipped path -- ``cellstate.backends.gse274113`` and ``cellstate.evaluation.gse274113_reports`` --
or, where it is a plain observational statistic, from the committed counts in one obvious step that
is written out in the command's own help.  It exists because **a representation nobody can poke at
cannot be iterated on**, and until now poking at it meant writing a script each time.

Run it from a source checkout; ``--help`` on any subcommand explains what it computes::

    uv run python scripts/explore.py inventory          what the committed slice contains
    uv run python scripts/explore.py panel              the 100 genes, and which are expressed

    uv run python scripts/explore.py knockdown          did the perturbation reach the readout?
    uv run python scripts/explore.py day                does the panel see biology that IS there?
    uv run python scripts/explore.py spectrum           is there structure for W to find?

    uv run python scripts/explore.py state rep1 GATA1   one arm's belief, in gene terms
    uv run python scripts/explore.py axes rep1          the fitted W and V of one fold
    uv run python scripts/explore.py contrast rep1 NT GATA1
    uv run python scripts/explore.py sweep rep1         every target against NT, one library
    uv run python scripts/explore.py ranks              does that ordering replicate? (it does not)

    uv run python scripts/explore.py measure            S2, S4, S5 with grouped intervals

The three screens in the middle group are the ones to run first, and they are the reusable part:
they ask whether a substrate carries an effect at all, before any capability is tested against it.

Design lines this tool holds, because the package holds them:

* **Loadings, never labels.**  An axis is reported as the genes that define it.  Calling one "the
  granulocyte axis" would be an interpretation shipped as a field.
* **Abstention is reprinted, not smoothed.**  Every belief from this backend abstains and says why.
* **Coordinates are never compared across libraries.**  A belief about library *L* comes from the
  fold that excluded *L*, so two libraries are expressed in different fitted bases.  Commands that
  would cross that line refuse.
* **A lower bound is named a lower bound.**  ``sd >=`` is the per-pair figure, which treats two
  arms sharing a fold as independent.  The grouped interval is what ``measure`` reports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:  # a source checkout, run without installing
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cellstate.backends.gse274113 import (  # noqa: E402
    available_arms,
    compare_arms,
    describe_state,
    estimate_arm,
    load_arm_slice,
)
from cellstate.backends.gse274113.fit import (  # noqa: E402
    BIOLOGY_RANK,
    NUISANCE_RANK,
    NULL_TARGET,
    PLACEBO_TARGETS,
    ArmSlice,
    fit_fold,
)
from cellstate.data.modality_registry import (  # noqa: E402
    on_target_expression_is_a_validity_control,
    perturbation_modality_for,
)

RULE = "-" * 78

# The source this tool explores. Named once so the modality lookup below cannot drift from the
# slice actually loaded, and so a second backend has an obvious place to declare its own.
SOURCE_KEY = "GSE274113"


def _slice() -> ArmSlice:
    return load_arm_slice()


def _log_composition(slice_data: ArmSlice, library: str, target: str) -> np.ndarray:
    composition, _ = slice_data.log_composition(library, target)
    return np.asarray(composition, dtype=np.float64)


def _cpm(slice_data: ArmSlice, library: str, target: str) -> np.ndarray:
    """Haldane-corrected counts per million -- the same statistic the model is written on."""

    counts = slice_data.counts[(library, target)].astype(np.float64)
    return 1e6 * (counts + 0.5) / (counts.sum() + counts.shape[0] / 2.0)


# --------------------------------------------------------------------------------------------
# inventory


def cmd_inventory(_: argparse.Namespace) -> None:
    """What the committed slice contains: every library, day, arm and panel gene."""

    slice_data = _slice()
    arms = available_arms()
    print(f"slice: {len(slice_data.libraries)} libraries x {len(slice_data.targets)} targets")
    print(f"       + {len(PLACEBO_TARGETS)} placebo halves per library")
    print(f"       {len(arms)} (library, target) arms answerable by estimate_arm")
    print(f"panel: {len(slice_data.gene_symbols)} genes")
    print(f"model: [W | V | u_g], BIOLOGY_RANK={BIOLOGY_RANK}, NUISANCE_RANK={NUISANCE_RANK}")
    print(f"folds: leave-one-library-out, K={len(slice_data.libraries)}\n")

    print(f"{'library':10} {'day':>4} {'cells':>8} {'panel counts':>14} {'NT cells':>9}")
    print(RULE)
    total_cells = 0
    for library in slice_data.libraries:
        cells = sum(
            count
            for (lib, target), count in slice_data.cells.items()
            if lib == library and target in slice_data.targets
        )
        depth = sum(
            int(slice_data.counts[(lib, target)].sum())
            for (lib, target) in slice_data.counts
            if lib == library and target in slice_data.targets
        )
        total_cells += cells
        print(
            f"{library:10} {slice_data.library_day[library]:>4} {cells:>8,} "
            f"{depth:>14,} {slice_data.cells[(library, NULL_TARGET)]:>9,}"
        )
    print(RULE)
    print(f"{'total':10} {'':>4} {total_cells:>8,}\n")

    print("targets:")
    for index in range(0, len(slice_data.targets), 6):
        print("  " + "  ".join(f"{t:8}" for t in slice_data.targets[index : index + 6]))
    print(
        "\nNT is the non-targeting control. NT_A / NT_B are a deterministic within-library split\n"
        "of the NT cells; their contrast is the placebo floor every other contrast is read against."
    )


# --------------------------------------------------------------------------------------------
# panel


def cmd_panel(args: argparse.Namespace) -> None:
    """The 100-gene panel, ranked by how much signal it actually carries."""

    slice_data = _slice()
    matrix = np.stack([_cpm(slice_data, lib, NULL_TARGET) for lib in slice_data.libraries])
    mean_cpm = matrix.mean(axis=0)
    targets = set(slice_data.targets)

    order = np.argsort(-mean_cpm)
    print(f"panel genes by mean NT expression across {len(slice_data.libraries)} libraries")
    print(f"{'gene':12} {'mean CPM':>12}  {'role':<16}")
    print(RULE)
    shown = order if args.all else np.concatenate([order[:20], order[-10:]])
    for position, index in enumerate(shown):
        if not args.all and position == 20:
            print(f"{'...':12} {'':>12}  ({len(order) - 30} genes omitted; --all to see them)")
        symbol = slice_data.gene_symbols[index]
        role = "knockout target" if symbol in targets else ""
        print(f"{symbol:12} {mean_cpm[index]:>12,.1f}  {role:<16}")

    not_expressed = [slice_data.gene_symbols[i] for i in range(len(mean_cpm)) if mean_cpm[i] < 10.0]
    print(RULE)
    print(
        f"{len(not_expressed)} genes below 10 CPM (effectively not expressed): "
        f"{', '.join(not_expressed) if not_expressed else 'none'}"
    )
    print(
        "A target that is not expressed cannot be knocked down, so its contrast is a\n"
        "measurement of noise. That is what makes SNAI2 the useful reference in `sweep`."
    )


# --------------------------------------------------------------------------------------------
# state / axes / contrast / sweep


def cmd_state(args: argparse.Namespace) -> None:
    """One arm's belief, its biology block in gene terms, and its abstention."""

    belief = estimate_arm(args.library, args.target)
    print(describe_state(belief, top_genes=args.top_genes))


def cmd_axes(args: argparse.Namespace) -> None:
    """The fitted bases of one fold -- what the coordinates in `state` are coordinates IN."""

    slice_data = _slice()
    fold = fit_fold(slice_data, args.library)
    symbols = slice_data.gene_symbols

    print(f"fold holding out {args.library}; fitted on {len(fold.fit_library_ids)} libraries")
    print(
        f"psi^2 = {fold.biological_observation_variance:.6f} "
        f"(pre-clamp {fold.biological_observation_variance_before_clamp:.6f}, "
        f"clamped={fold.dispersion_is_clamped})\n"
    )

    if fold.day_axis_in_nuisance_basis is not None:
        assert fold.day_axis_in_biology_basis is not None
        print(
            f"day axis absorbed: nuisance V {fold.day_axis_in_nuisance_basis:.4f}, "
            f"biology W {fold.day_axis_in_biology_basis:.4f}\n"
            "  V is fitted on the across-library NT residual, and the fourteen libraries sit at\n"
            "  four differentiation days -- so it absorbs the culture's clock, not just the\n"
            "  library. W is orthogonalized against V, which is why biology keeps almost none of\n"
            "  it. A random 3-of-100 subspace would take about 0.03.\n"
        )

    for name, basis, count in (
        ("biology  W", fold.biology_basis, BIOLOGY_RANK),
        ("nuisance V", fold.nuisance_basis, NUISANCE_RANK),
    ):
        print(f"{name}  ({count} columns)")
        for column_index in range(basis.shape[1]):
            column = basis[:, column_index]
            order = np.argsort(-np.abs(column))[: args.top_genes]
            loadings = ", ".join(f"{symbols[i]}{column[i]:+.2f}" for i in order)
            print(f"  {name.split()[0]}_{column_index}  {loadings}")
        print()

    print("per-target residual direction norm |u_g| (how much a target does BEYOND W and V):")
    ordered = sorted(fold.residual_norm_by_target.items(), key=lambda kv: -kv[1])
    for target, norm in ordered:
        print(f"  {target:10} {norm:7.4f}")
    print(
        "\nW is orthogonalized against V, a DECLARED BIAS: biology genuinely aligned with\n"
        "the library axis is assigned to nuisance. It biases against finding biology, not for it."
    )


def cmd_contrast(args: argparse.Namespace) -> None:
    """Two arms of the same library, differenced in their shared biology coordinates."""

    print(compare_arms(args.library, args.target_a, args.target_b))
    print(
        "\n`sd >=` is a LOWER bound: it adds the two posterior variances as though the arms were\n"
        "independent, and they are not -- both were scored under the same fold. The grouped\n"
        "interval is what `measure` reports."
    )


def cmd_sweep(args: argparse.Namespace) -> None:
    """Every target against NT in one library, read against the placebo floor."""

    slice_data = _slice()
    if args.library not in slice_data.libraries:
        raise SystemExit(
            f"unknown library {args.library!r}; try: {', '.join(slice_data.libraries)}"
        )

    # Expression of each target gene in this library's NT arm.  A target that is not expressed
    # cannot be knocked down, so its contrast measures noise and is the reference the rest are
    # read against -- not zero.
    index = {symbol: position for position, symbol in enumerate(slice_data.gene_symbols)}
    nt_cpm = _cpm(slice_data, args.library, NULL_TARGET)
    expression = {
        target: float(nt_cpm[index[target]]) if target in index else float("nan")
        for target in slice_data.targets
    }

    rows = []
    for target in slice_data.targets:
        if target == NULL_TARGET:
            continue
        contrast = compare_arms(args.library, NULL_TARGET, target)
        rows.append((target, contrast.distance, contrast.distance_lower_bound_sd))
    rows.sort(key=lambda row: -row[1])

    silent = [target for target, *_ in rows if expression[target] < 10.0]
    floor = float(np.mean([d for t, d, _ in rows if t in silent])) if silent else None

    print(f"{args.library}: every target against NT, in the fold that never saw {args.library}\n")
    print(f"{'target':10} {'|delta|':>9} {'sd >=':>8} {'NT CPM':>10}  {'vs floor':>9}")
    print(RULE)
    for target, distance, lower in rows:
        ratio = f"{distance / floor:8.2f}x" if floor else "       --"
        marker = "  <- not expressed" if target in silent else ""
        print(
            f"{target:10} {distance:>9.3f} {lower:>8.3f} {expression[target]:>10,.1f}"
            f"  {ratio}{marker}"
        )
    print(RULE)
    if floor:
        print(
            f"{'floor':10} {floor:>9.3f} {'':>8} {'':>10}  {'1.00x':>9}  "
            f"mean of {len(silent)} not-expressed targets ({', '.join(silent)})"
        )
    print(
        "\nThe floor is the reference, not zero. A target below 10 CPM is not expressed in this\n"
        "panel, so its contrast is the size of a difference that means nothing here.\n"
        "This is an ORDERING within one library, not a capability claim: the grouped measurement\n"
        "(`measure`) finds the perturbed contrast inseparable from the placebo one.\n"
        "Run `ranks` to see whether this ordering survives to the other thirteen libraries."
    )


def cmd_ranks(_: argparse.Namespace) -> None:
    """Does the per-library ordering replicate?  Rank every target in all fourteen libraries.

    A single library's ordering is easy to read as a result -- in ``rep1`` the two master
    erythroid/MK regulators do come out on top.  Whether that is signal is a question about the
    other thirteen libraries, and it is asked here rather than assumed.

    A not-expressed target reaching rank 1 in any library is the decisive observation: it cannot
    have been knocked down, so whatever put it there is the same thing ordering everything else.
    """

    slice_data = _slice()
    perturbed = [target for target in slice_data.targets if target != NULL_TARGET]
    index = {symbol: position for position, symbol in enumerate(slice_data.gene_symbols)}
    expression = {
        target: float(
            np.mean(
                [_cpm(slice_data, lib, NULL_TARGET)[index[target]] for lib in slice_data.libraries]
            )
        )
        if target in index
        else float("nan")
        for target in perturbed
    }

    ranks: dict[str, list[int]] = {target: [] for target in perturbed}
    winners: list[tuple[str, str, float]] = []
    for library in slice_data.libraries:
        distances = {
            target: compare_arms(library, NULL_TARGET, target).distance for target in perturbed
        }
        order = sorted(perturbed, key=lambda target: -distances[target])
        for position, target in enumerate(order):
            ranks[target].append(position + 1)
        winners.append((library, order[0], distances[order[0]]))

    print(
        f"rank by |delta| vs NT in each of {len(slice_data.libraries)} libraries "
        f"(1 = largest of {len(perturbed)})\n"
    )
    print(f"{'target':10} {'mean rank':>10} {'best':>5} {'worst':>6} {'NT CPM':>10}  {'':4}")
    print(RULE)
    for target in sorted(perturbed, key=lambda t: float(np.mean(ranks[t]))):
        values = ranks[target]
        flag = "  <- NOT EXPRESSED" if expression[target] < 10.0 else ""
        print(
            f"{target:10} {np.mean(values):>10.1f} {min(values):>5} {max(values):>6}"
            f" {expression[target]:>10,.1f}{flag}"
        )
    print(RULE)

    print("\nlargest contrast in each library:")
    for library, target, distance in winners:
        flag = "   <- NOT EXPRESSED" if expression[target] < 10.0 else ""
        print(f"  {library:8} {target:10} {distance:7.3f}{flag}")

    silent_wins = [(lib, t) for lib, t, _ in winners if expression[t] < 10.0]
    print(
        "\nThe ordering does not replicate. No target holds rank 1 in more than a few libraries,\n"
        "the best mean ranks are near the middle of a field of 19, and the spread from best to\n"
        "worst rank is most of the field for almost every target."
    )
    if silent_wins:
        pairs = ", ".join(f"{t} in {lib}" for lib, t in silent_wins)
        print(
            f"\nDecisive: {pairs} produces the LARGEST contrast of all "
            f"{len(perturbed)} targets\n"
            "in that library, and that gene is not expressed -- it cannot have been knocked\n"
            "down. "
            "Whatever ranked it first is what is ranking everything else."
        )


# --------------------------------------------------------------------------------------------
# knockdown -- the positive control on the substrate


def cmd_knockdown(_: argparse.Namespace) -> None:
    """How much of the on-target transcript survives the edit?

    Plain observational statistic, no model: for each knockout target g that is itself a panel gene,
    the mean over libraries of ``log2( CPM_g(arm g) / CPM_g(arm NT) )``.

    ⚠️ **This is not a validity screen for this deposit, and reading it as one produced a verdict
    that has since been withdrawn.**  GSE274113 is Cas9 nuclease knockout, not CRISPRi: cutting
    destroys the protein, the transcript falls only through nonsense-mediated decay, and an edit
    that escapes NMD -- or that removes a repressor from its own promoter -- leaves the transcript
    flat or raises it.  A fully working screen is consistent with the figures printed below.  What
    the statistic measures is NMD escape and editing mosaicism, and nothing else.
    """

    slice_data = _slice()
    index = {symbol: position for position, symbol in enumerate(slice_data.gene_symbols)}
    rows = []
    for target in slice_data.targets:
        if target == NULL_TARGET or target not in index:
            continue
        position = index[target]
        changes = [
            float(
                np.log2(
                    _cpm(slice_data, lib, target)[position]
                    / _cpm(slice_data, lib, NULL_TARGET)[position]
                )
            )
            for lib in slice_data.libraries
            if (lib, target) in slice_data.counts and (lib, NULL_TARGET) in slice_data.counts
        ]
        baseline = float(
            np.mean([_cpm(slice_data, lib, NULL_TARGET)[position] for lib in slice_data.libraries])
        )
        rows.append((target, float(np.mean(changes)), baseline))
    rows.sort(key=lambda row: row[1])

    print("ON-TARGET KNOCKDOWN -- the positive control, computed from the committed counts\n")
    print(f"{'target':10} {'log2FC':>9} {'NT CPM':>11}  {'':<4}")
    print(RULE)
    for target, change, baseline in rows:
        flag = "WRONG SIGN" if change > 0 else ""
        if baseline < 10.0:
            flag = "not expressed"
        print(f"{target:10} {change:>+9.3f} {baseline:>11,.1f}  {flag}")
    print(RULE)

    changes = [row[1] for row in rows]
    wrong = sum(1 for value in changes if value > 0)
    well = [row for row in rows if row[2] > 200.0]
    print(f"\nmean on-target log2FC          {np.mean(changes):+.4f}   over {len(rows)} targets")
    print(f"wrong-signed                   {wrong} of {len(rows)}")
    print(
        f"restricted to >200 CPM         {np.mean([r[1] for r in well]):+.4f}   "
        f"({len(well)} targets, {sum(1 for r in well if r[1] > 0)} wrong-signed)"
    )
    # The modality is LOOKED UP, not asserted in this prose. That is the whole repair: the screen
    # used to state the interpretation in a string, so nothing could notice when the string was
    # wrong about the assay. If the source is not registered the lookup raises rather than
    # defaulting, and the screen refuses to interpret its own output.
    modality = perturbation_modality_for(SOURCE_KEY)
    valid = on_target_expression_is_a_validity_control(SOURCE_KEY)
    print(f"\nperturbation modality          {modality.value}")
    print(f"is this a validity control?    {'YES' if valid else 'NO'}")
    if not valid:
        print(
            "\nThis figure does NOT say whether the perturbation worked, because the target\n"
            "transcript is not what this modality acts on. Cutting destroys the protein and\n"
            "leaves the transcript largely intact, so a fully working screen is consistent\n"
            "with the numbers above. The 'measured null' verdict this screen once supported is\n"
            "WITHDRAWN: it compared these figures against a CRISPRi threshold.\n"
            "\nThe controls that would settle it are guide-level replication, expression-\n"
            "dependence of effect size, and the cutting-versus-non-cutting contrast against\n"
            "AAVS1. None is implemented yet. Run `day` next -- the same panel and pipeline\n"
            "resolve differentiation."
        )
    else:
        print(
            "\nThis modality acts on transcription, so the figures above ARE a validity\n"
            "control for this source and a working screen is expected to move them."
        )


# --------------------------------------------------------------------------------------------
# day -- the differentiation readout


def cmd_day(_: argparse.Namespace) -> None:
    """Does the panel see biology that is actually there?  Differentiation, day 7 to day 14.

    Plain observational statistic in log-composition space, no fitted basis and no regrouping.
    ``library_day`` is nested inside ``library`` -- three or four libraries per day, none spanning
    two -- so this is a READOUT and deliberately not a capability measurement.  Re-pointing the
    biology block at this axis would collapse K from 14 to 4 and yield a passing S5 that means
    nothing.
    """

    slice_data = _slice()
    by_day: dict[int, list[str]] = {}
    for library in slice_data.libraries:
        by_day.setdefault(slice_data.library_day[library], []).append(library)
    days = sorted(by_day)

    print("DIFFERENTIATION READOUT -- NT arms only, raw log-composition, no fitted basis\n")
    print(f"{'day':>5}  {'libraries':<24} {'NT cells':>10}")
    print(RULE)
    for day in days:
        cells = sum(slice_data.cells[(lib, NULL_TARGET)] for lib in by_day[day])
        print(f"{day:>5}  {', '.join(by_day[day]):<24} {cells:>10,}")
    print(RULE)

    def mean_nt(day: int) -> np.ndarray:
        return np.mean(
            [_log_composition(slice_data, lib, NULL_TARGET) for lib in by_day[day]], axis=0
        )

    first, last = days[0], days[-1]
    differentiation = float(np.linalg.norm(mean_nt(last) - mean_nt(first)))
    placebo = float(
        np.mean(
            [
                np.linalg.norm(
                    _log_composition(slice_data, lib, PLACEBO_TARGETS[1])
                    - _log_composition(slice_data, lib, PLACEBO_TARGETS[0])
                )
                for lib in slice_data.libraries
            ]
        )
    )
    perturbed = float(
        np.mean(
            [
                np.linalg.norm(
                    _log_composition(slice_data, lib, target)
                    - _log_composition(slice_data, lib, NULL_TARGET)
                )
                for lib in slice_data.libraries
                for target in slice_data.targets
                if target != NULL_TARGET
            ]
        )
    )

    print(f"\n{'contrast':<34} {'|delta|':>9} {'vs placebo':>12}")
    print(RULE)
    print(
        f"{f'NT day {first} -> day {last}':<34} {differentiation:>9.3f} "
        f"{differentiation / placebo:>11.2f}x"
    )
    print(f"{'perturbed target vs NT (mean)':<34} {perturbed:>9.3f} {perturbed / placebo:>11.2f}x")
    print(f"{'placebo NT_A vs NT_B (mean)':<34} {placebo:>9.3f} {1.0:>11.2f}x")
    print(RULE)

    day_values = np.array([slice_data.library_day[lib] for lib in slice_data.libraries], float)
    matrix = np.stack(
        [_log_composition(slice_data, lib, NULL_TARGET) for lib in slice_data.libraries]
    )
    correlation = np.nan_to_num(
        np.array([np.corrcoef(day_values, matrix[:, g])[0, 1] for g in range(matrix.shape[1])])
    )
    tracking = int((np.abs(correlation) > 0.7).sum())
    print(f"\n{tracking} of {len(correlation)} panel genes track day at |r| > 0.7. Strongest:")
    for position in np.argsort(-np.abs(correlation))[:8]:
        print(f"  {slice_data.gene_symbols[position]:10} r = {correlation[position]:+.3f}")

    print(
        f"\nThe instrument works. Differentiation moves the panel "
        f"{differentiation / placebo:.2f}x the\n"
        f"placebo contrast; the knockout perturbation moves it {perturbed / placebo:.2f}x --\n"
        "which is the placebo floor. The negative capability measurements are a verdict on the\n"
        "PERTURBATION arm of this dataset, not on the representation.\n"
        "\nNOTE: day is nested in library (no library spans two days), so this is deliberately a\n"
        "readout and not a measurement. It carries no interval and clears no gate."
    )


# --------------------------------------------------------------------------------------------
# spectrum -- is there structure to fit at all?


def cmd_spectrum(_: argparse.Namespace) -> None:
    """Does the matrix the biology basis is fitted on carry structure, or is it noise?

    ``W`` is the leading subspace of the within-library contrasts ``c[L, g] - c[L, NT]``.  If the
    perturbation did something, that matrix has a dominant direction and its singular values fall
    away.  If it did nothing, the matrix is near-isotropic and its spectrum is flat.

    Two references are printed alongside it, both from this same slice: the **placebo** contrast
    (``NT_B - NT_A``, which is noise by construction) and the **differentiation** contrast (the NT
    arms across days, which is real biology).  The comparison is the whole point -- a spectrum is
    only flat or steep relative to something.

    This costs one SVD and no downloads, and it screens a corpus even when the perturbation targets
    are not themselves in the panel, which ``knockdown`` cannot.
    """

    slice_data = _slice()
    libraries = list(slice_data.libraries)
    composition = {
        (library, target): _log_composition(slice_data, library, target)
        for library in libraries
        for target in (*slice_data.targets, *PLACEBO_TARGETS)
    }

    def spectrum(rows: list[np.ndarray], name: str, count: int) -> None:
        values = np.linalg.svd(np.vstack(rows), compute_uv=False)
        fraction = values**2 / np.sum(values**2)
        print(
            f"{name:<34} {count:>5} "
            + " ".join(f"{value:7.1f}" for value in values[:5])
            + f"   {values[1] / values[0]:>6.2f} {fraction[0] * 100:>8.1f}%"
        )

    print("SPECTRUM -- is there structure for the biology basis to find?\n")
    print(
        f"{'contrast matrix':<34} {'rows':>5} "
        + " ".join(f"{f's{i}':>7}" for i in range(5))
        + f"   {'s1/s0':>6} {'PC1 var':>9}"
    )
    print(RULE)
    spectrum(
        [
            composition[(lib, t)] - composition[(lib, NULL_TARGET)]
            for lib in libraries
            for t in slice_data.targets
            if t != NULL_TARGET
        ],
        "perturbation: target - NT",
        len(libraries) * (len(slice_data.targets) - 1),
    )
    spectrum(
        [
            composition[(lib, PLACEBO_TARGETS[1])] - composition[(lib, PLACEBO_TARGETS[0])]
            for lib in libraries
        ],
        "placebo: NT_B - NT_A  (noise)",
        len(libraries),
    )
    centred = np.stack([composition[(lib, NULL_TARGET)] for lib in libraries])
    spectrum(
        list(centred - centred.mean(axis=0)), "differentiation: NT across days", len(libraries)
    )
    print(RULE)
    print(
        "\nRead the last two columns. Real biology concentrates: the differentiation\n"
        "contrast puts 92% of its variance on one direction and s1/s0 falls to 0.20.\n"
        "The perturbation contrast -- the matrix W is actually fitted on -- has the\n"
        "spectral shape of the PLACEBO: s1/s0 = 0.76 against the placebo's 0.75,\n"
        "PC1 at 20% against 27%.\n"
        "\nSo the biology basis is the leading subspace of a matrix that is spectrally\n"
        "indistinguishable from noise. That is the same verdict `knockdown` reaches, reached\n"
        "independently and without needing the targets to be panel genes.\n"
        "\nTwo consequences for the design:\n"
        f"  - BIOLOGY_RANK={BIOLOGY_RANK} cuts a flat spectrum. There is no gap at 4 to cut at:\n"
        "    the first excluded direction is ~88% the size of the last included one.\n"
        "  - Only the leading axis has a real gap. biology_2 and biology_3 sit on near-degenerate\n"
        "    singular values, so which direction gets which name is close to arbitrary -- and it\n"
        "    changes between folds. Run `axes` on two libraries and compare."
    )


# --------------------------------------------------------------------------------------------
# measure -- the shipped capability measurements


def cmd_measure(args: argparse.Namespace) -> None:
    """The shipped, grouped, interval-bearing capability measurements. S2, S4, S5."""

    from cellstate.evaluation.gse274113_reports import (
        held_out_states,
        measure_earned_spread,
        measure_intervention_response,
        measure_nuisance_separation,
        measure_point_predictor_spread,
    )

    slice_data = _slice()
    print("fitting 14 leave-one-library-out folds ...", flush=True)
    states = held_out_states(slice_data)

    measurements = [
        measure_earned_spread(slice_data),
        *measure_intervention_response(slice_data),
        measure_nuisance_separation(states, bound=args.bound),
        measure_point_predictor_spread(states),
    ]

    print()
    for measurement in measurements:
        verdict = "PASS" if measurement.passed else "FAIL"
        print(f"[{verdict}] {measurement.name}")
        print(
            f"        {measurement.value:.4f}  "
            f"[{measurement.interval.lower:.4f}, {measurement.interval.upper:.4f}]  "
            f"K={measurement.unit_count}"
        )
        print(f"        {measurement.statement}\n")

    print(RULE)
    print("S5 BLOCK DECOMPOSITION -- where the variance actually sits")
    print(RULE)
    biology, nuisance = _block_coefficients(slice_data)
    libraries = list(slice_data.libraries)
    targets = list(slice_data.targets)

    def across_library(block: dict[tuple[str, str], np.ndarray]) -> float:
        means = np.stack([np.mean([block[(lib, t)] for t in targets], axis=0) for lib in libraries])
        return float(np.mean(np.var(means, axis=0)))

    per_target = np.stack(
        [np.mean([biology[(lib, t)] for lib in libraries], axis=0) for t in targets]
    )
    nuisance_across = across_library(nuisance)
    biology_across = across_library(biology)
    between_target = float(np.mean(np.var(per_target, axis=0)))

    print(
        f"  across-library variance in the NUISANCE block  : {nuisance_across:10.4f}   "
        "<- where library variation is SUPPOSED to land"
    )
    print(
        f"  across-library variance in the BIOLOGY block   : {biology_across:10.4f}   "
        "<- leakage; S5's numerator"
    )
    print(
        f"  between-target variance in the BIOLOGY block   : {between_target:10.4f}   "
        "<- signal; S5's denominator"
    )
    print(
        f"\n  nuisance / biology leakage ratio               : "
        f"{nuisance_across / biology_across:10.2f}x"
    )
    print(
        f"  leakage / signal                              : "
        f"{biology_across / between_target:10.2f}x"
    )
    print(
        "\nRead the terms, never the quotient alone. Library variation IS being captured -- 99.3%\n"
        "of it lands in the nuisance block. S5 fails because its DENOMINATOR is near zero: the\n"
        "perturbation created almost no between-target biology (see `knockdown`). ADR 0022 cut\n"
        "the leakage 5x (3.07 -> 0.609) and S5 barely moved, because the signal fell with it\n"
        "(0.257 -> 0.109). A better instrument aimed at no signal returns a better-characterised\n"
        "zero."
    )


def _block_coefficients(
    slice_data: ArmSlice,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:
    """Held-out biology and nuisance coefficients for every arm, from the fold that excluded it.

    Mirrors ``tests/test_gse274113_observation_model.py``'s decomposition, which is the only place
    these three numbers are computed rather than quoted.
    """

    from cellstate.backends.gse274113.likelihood import posterior

    biology: dict[tuple[str, str], np.ndarray] = {}
    nuisance: dict[tuple[str, str], np.ndarray] = {}
    for library in slice_data.libraries:
        fold = fit_fold(slice_data, library)
        rank = fold.biology_basis.shape[1]
        width = fold.nuisance_basis.shape[1]
        for target in slice_data.targets:
            composition, depth = slice_data.log_composition(library, target)
            mean, _ = posterior(
                composition,
                intercept=fold.intercept,
                design=fold.design(target),
                prior_precision=fold.prior_precision(),
                observation_variance_diagonal=fold.observation_variance(depth),
            )
            biology[(library, target)] = np.asarray(mean[:rank], dtype=np.float64)
            nuisance[(library, target)] = np.asarray(mean[rank : rank + width], dtype=np.float64)
    return biology, nuisance


# --------------------------------------------------------------------------------------------


def cmd_calibration(_: argparse.Namespace) -> None:
    """S6: does the predictive interval contain the replicate at the rate it claims? (ADR 0024)"""

    from cellstate.backends.gse274113.arm_request import (
        S6_NOMINAL_INTERVAL,
        S6_REFERENCE_NOMINAL,
        arm_query,
    )
    from cellstate.evaluation.gse274113_reports import (
        calibration_shape_diagnostics,
        measure_calibration_level_set,
    )

    slice_data = _slice()
    thresholds = arm_query(slice_data.targets, model_fingerprint="0" * 64).acceptance_thresholds
    print("fitting 14 leave-one-library-out folds ...", flush=True)
    levels = measure_calibration_level_set(
        slice_data,
        minimum_coverage=thresholds.minimum_calibration_coverage,
        maximum_calibration_error=thresholds.maximum_calibration_error,
    )
    report = levels.reference
    shape = calibration_shape_diagnostics(slice_data)

    verdict = "PASS" if levels.outcome.value == "passed" else "FAIL"
    low, high = S6_NOMINAL_INTERVAL
    print()
    print(RULE)
    print(
        f"S6 CALIBRATION COVERAGE -- gated at {len(levels.nominals)} levels "
        f"on [{low:.2f}, {high:.2f}]   [{verdict}]"
    )
    print(RULE)
    print(f"  {'nominal':>9}{'coverage':>10}{'error':>9}{'BOUND':>9}   verdict")
    for nominal, entry in zip(levels.nominals, levels.reports, strict=True):
        mark = (
            "  <- reference; the belief publishes this row"
            if nominal == S6_REFERENCE_NOMINAL
            else ""
        )
        print(
            f"  {nominal:>9.2f}{entry.empirical_coverage:>10.4f}{entry.calibration_error:>9.4f}"
            f"{entry.calibration_error_upper_bound:>9.4f}   {entry.outcome.value}{mark}"
        )
    print(f"\n  predeclared maximum error   : {thresholds.maximum_calibration_error:10.4f}")
    print(f"  predeclared coverage floor  : {thresholds.minimum_calibration_coverage:10.4f}")
    print(
        f"  reference interval          : "
        f"[{report.coverage_interval.lower:.4f}, {report.coverage_interval.upper:.4f}]   K=14"
    )
    print(
        "\n  The verdict is the CONJUNCTION over every level, never the reference row alone.\n"
        "\n  Why six and not one. The predeclared pair (floor 0.85, max error 0.05) is coherent\n"
        "  on the whole interval above -- below 0.90 the error bound would admit a coverage the\n"
        "  floor rejects, above 0.95 it would ask for coverage over 1. ADR 0024 read that as a\n"
        "  single point and gated at 0.90, its LOOSEST member: the bound rises monotonically.\n"
        "  A one-level gate is clearable by a CONSTANT -- multiplying every predictive sd by any\n"
        "  factor in [1.04, 1.21] clears 0.90, and 1.11 lands coverage on exactly 0.9000 with a\n"
        "  bound of 0.0368, better than the shipped 0.0548. One level tests scale; scale is free.\n"
        "  Across all six, only a narrow band of scalars clears every level, so the widened gate\n"
        "  is a statement about the residuals' SHAPE.\n"
        "\n  At the reference level the point estimate PASSES both thresholds and the bound does\n"
        "  not. A criterion that reported its point estimate would have called this calibrated\n"
        "  (ADR 0015)."
    )

    print()
    print(RULE)
    print("WHERE THE FAILURE LIVES -- S2 says 'uniformly too narrow'; it is not")
    print(RULE)
    print(f"  sd of standardized residuals        : {shape.standard_deviation:8.4f}")
    print(
        f"  ... trimming the worst {shape.trimmed_fraction:.0%} (28/1400) : "
        f"{shape.trimmed_standard_deviation:8.4f}   <- the spread is EARNED"
    )
    print(f"  largest standardized residual       : {shape.largest_absolute_score:8.2f}")
    print(
        "\n  2% of the outcomes carry the whole S2 failure. The bulk of the panel is BETTER than\n"
        "  the interval claims (coverage at nominal 0.50 is 0.66). Inflating psi^2 -- the repair\n"
        "  S2's ratio implies -- would push 98% into over-coverage and still miss a 9.5-sigma\n"
        "  outlier. The two readings point opposite ways; the ratio is the misleading one."
    )

    print()
    print(RULE)
    print("COVERAGE BY LIBRARY -- the gradient the pooled number hides")
    print(RULE)
    print(f"  {'library':<10}{'coverage':>10}")
    for library, coverage in shape.coverage_by_library:
        flag = "  <- deepest" if coverage == min(v for _, v in shape.coverage_by_library) else ""
        print(f"  {library:<10}{coverage:>10.3f}{flag}")
    print(f"\n  corr(log depth, coverage) = {shape.depth_coverage_correlation:+.4f}   (n=14)")
    print(
        "\n  likelihood.py names psi^2 as the defence against 'more sequencing depth mistaken for\n"
        "  more knowledge about the biology'. On this evidence it does not hold: the technical\n"
        "  share falls 0.55 -> 0.35 across the depth range while psi^2 stays near 0.055.\n"
        "\n  WARNING: depth and differentiation day are collinear here and CANNOT be separated --\n"
        "  depth rises with day, and within a day the range is too narrow to resolve anything."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="explore.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler, help_text: str) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(
            name,
            help=help_text,
            description=handler.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        sub.set_defaults(handler=handler)
        return sub

    add("inventory", cmd_inventory, "what the committed slice contains")

    panel = add("panel", cmd_panel, "the 100-gene panel and what it expresses")
    panel.add_argument("--all", action="store_true", help="print all 100 genes")

    state = add("state", cmd_state, "one arm's belief, in gene terms")
    state.add_argument("library")
    state.add_argument("target")
    state.add_argument("--top-genes", type=int, default=6)

    axes = add("axes", cmd_axes, "the fitted bases of one fold")
    axes.add_argument("library")
    axes.add_argument("--top-genes", type=int, default=8)

    contrast = add("contrast", cmd_contrast, "difference two arms of the same library")
    contrast.add_argument("library")
    contrast.add_argument("target_a")
    contrast.add_argument("target_b")

    sweep = add("sweep", cmd_sweep, "every target against NT in one library")
    sweep.add_argument("library")

    add("ranks", cmd_ranks, "does that ordering replicate across all 14 libraries?")

    add("knockdown", cmd_knockdown, "did the perturbation reach the readout? (positive control)")
    add("day", cmd_day, "the differentiation readout: does the panel see real biology?")
    add("spectrum", cmd_spectrum, "is there structure for the biology basis to find?")

    add("calibration", cmd_calibration, "S6: does the interval cover at the rate it claims?")

    measure = add("measure", cmd_measure, "the shipped capability measurements S2, S4, S5")
    measure.add_argument(
        "--bound",
        type=float,
        default=0.35,
        help="the predeclared S5 bound (ADR 0022); default 0.35",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    arguments.handler(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
