# Prelievo completo: full-coverage harvest scheduling

Spec for the `prelievo_completo` option on `@@piano_di_taglio` and `@@prelievi`.
Status: design approved, not yet implemented.

## Problem

The harvest scheduler starves eligible parcels. In the 2027–2040 plan, five
fustaia parcels are never cut even though the harvest rules permit a cut in at
least one year:

| parcel | comparto | ha | peak vol/ha | floor | eligible years | harvest |
|---|---|---|---|---|---|---|
| Fabrizia/11a | D | 26.2 | 556 | 300 | 2035–2040 | 2,628–2,874 m³ |
| Capistrano/1 | C | 19.0 | 515 | 300 | 2035–2040 | 1,824–1,917 m³ |
| Capistrano/2 | C | 17.9 | 362 | 300 | 2036–2040 | 462–749 m³ |
| Capistrano/6a | B | 13.9 | 478 | 420 | 2035–2040 | 498–524 m³ |
| Capistrano/7a | C | 1.2 | 307 | 300 | 2039–2040 | 23–24 m³ |

Cause: each year the greedy loop stops once `volume_obiettivo` and
`particelle_min` are met, and priority is descending mature volume/ha
(`ORDINE_VOL_HA`). Because `intervallo` (10 years) is shorter than the plan
(14 years), the dense parcels come back off their rest period and re-take the
quota before the queue ever drains. It is a closed cycle, not a transient
delay: 13 parcels receive a *second* cut while these five receive none.

This contradicts `sec-metodo.tex`, which describes thinning as *"cauto e
capillare"*, and the prescribed treatment for four of the five parcels
specifies an actual intervention (*"Diradamento per creare spazi…"*,
*"Si interviene sui soggetti dominati e difettosi"*).

The parcel ordering itself is a deliberate silvicultural choice, documented in
`relazione.tex:278` — intervene on the densest stands first to reduce
competition and let sparser ones grow. **This design preserves that ordering
unchanged.** A parcel at 400 m³/ha must not pre-empt one at 1000 m³/ha.

## Scope

In scope: the scheduler option specified below.

Out of scope: the treatment of young comparto-A stands, where the volume
floor (120% of *provvigione minima*) currently vetoes any harvest —
Capistrano 12b/16b/17b and Fabrizia 11c. Whether those stands should be
governed by the basal-area limit alone, the volume limit, or both is under
review, and may require a per-parcel way to select which limits apply. That
question is independent of the scheduling defect specified here.

## Behaviour

New option `prelievo_completo`, si/no, default **no**.

- **absent or `no`** — current behaviour, byte-identical output.
- **`si`** — no parcel that the rules permit to be cut is left uncut at the
  end of the plan.

Recovered harvests are appended to the plan; the parcel ordering and the
commit rule are untouched. When no recovered parcel comes off its rest period
again before the plan ends — `intervallo` ≥ the years remaining after its
placement, which holds for every debt in the current data — the option is
strictly additive: the base schedule is reproduced exactly and the extra
harvests are added to it. Otherwise a recovered parcel may re-enter the normal
pool in a later year and shift what that year schedules; the coverage
guarantee still holds, but the base plan is no longer reproduced row for row.

Like every other plan parameter it must be given identically to
`@@piano_di_taglio` and `@@prelievi`, which share `parse_plan_params` and
must describe the same plan.

## Definitions

**Eligible.** Parcel *p* is eligible in year *y* iff *p* is fustaia, is not
resting (`last_harvest[p] <= y - gap(y)`), and `harvest_parcel` for *p* at
year *y* returns a result with `harvest > 0`.

The third clause is essential and is why the year loop can no longer stop
early: a parcel that was never reached might also have yielded zero had it
been reached, and only calling `harvest_parcel` distinguishes "starved" from
"rules declined". `harvest_parcel` does not mutate (the caller performs
`sim.drop`), so evaluating past the target is safe.

**Debt.** A parcel eligible in at least one year and committed in none.

**Debt year** `debt_year(p)` = the earliest year *p* is eligible.
**Debt size** `debt_size(p)` = `harvest(p, debt_year(p))`.

Debt size is measured at the debt year, not at the year eventually chosen.
A parcel's harvest grows with the year selected (Fabrizia/11a yields 2,677 m³
in 2036 and 2,874 m³ in 2040), so any size measured across the candidate
years would presuppose the placement decision it is being used to make.

## Placement

Assigning debts to years is greedy load balancing: the years are machines
carrying the base plan's volumes as pre-existing load, and each debt is a job
whose size depends on the machine chosen.

**Drain order** — debts are settled in the order they were incurred:

```
(debt_year asc, debt_size desc, compresa asc, natsort(particella) asc)
```

Size-descending is the secondary key, not the primary one. It earns its place
because simultaneous debts are common — Fabrizia/11a, Capistrano/1 and
Capistrano/6a all come due in 2035 — and placing the largest first stops a
big debt from arriving last and having to pile onto an already-levelled
profile.

**Placement year** — each debt goes to

```
argmin over y in eligible(p) of (annual_total[y], y)
```

where `annual_total` includes debts already placed. Ties resolve to the
earliest year. Because the drain order is total and the placement rule is
deterministic, the result is reproducible.

**Stickiness.** Once placed, a debt keeps its placement in all later
iterations. Only newly discovered debts are placed. This makes the forced set
append-only, which is what gives the termination bound below.

Empirically the drain order does not affect balance on the current data (all
candidate orders reach the same peak and standard deviation); it is specified
for determinism and explicability, not for optimality.

## Algorithm

```
placements = {}                       # parcel -> year, append-only
for iteration in 1 .. max_iterations:
    run = simulate(forced=placements)
    #   run.eligible  : (parcel, year) -> harvest, measured in this run
    #   run.committed : parcels cut in this run, by the normal loop or forced
    #   run.totals    : year -> volume harvested, this run
    debts = { p : p appears in run.eligible and p not in run.committed }
    if not debts:
        return run
    totals = copy(run.totals)         # running load, updated as debts are placed
    for p in sorted(debts, key=drain_order):
        y = argmin over eligible_years(p, run) of (totals[y], y)
        placements[p] = y
        totals[y] += run.eligible[(p, y)]
raise  # bound exceeded
```

Eligibility, harvest volumes and annual loads are all read from the run that
has just completed. A debt was cut in no year of that run, so its recorded
eligible set and harvest figures describe an uncut parcel and are exactly the
ones the placement needs.

`simulate` is the existing `schedule_harvests`, with its year loop amended as
follows:

```
for y in years:
    order = priority(parcels, y)          # unchanged: ORDINE_* as configured
    year_total, year_parcels, target_met = 0.0, 0, False
    for p in order:
        if gap_blocked(p, y):
            continue
        result = harvest_parcel(p, ...)   # no mutation
        if result is None or result.harvest == 0:
            continue
        if complete:
            eligible[(p, y)] = result.harvest
        if target_met:
            continue                      # eligible, deliberately not committed
        commit(p, y, result)
        year_total += result.harvest
        year_parcels += 1
        if year_total >= target_volume and year_parcels >= particelle_min:
            target_met = True
            if not complete:
                break                     # default path: loop shape unchanged
    for p in forced_at(y):                # after the loop, before growth
        commit_forced(p, y)               # assert harvest > 0
    growth_step()
```

Two invariants make this work:

1. **Forced harvests never enter `year_total`.** Forcing a debt in year *y*
   therefore cannot displace anything the normal loop scheduled in year *y*.
   It can still affect a *later* year, because the recovered parcel rejoins
   the normal pool once its rest period expires — which is precisely the case
   the fixpoint below exists to absorb.
2. **Forced harvests are applied after the parcel loop and before the growth
   step**, so removed trees do not grow, and the forced parcel's own trees
   are untouched by that year's other harvests (drops only affect the
   harvested parcel's rows). Committing a debt after the loop yields exactly
   what committing it mid-loop would have.

**Why a fixpoint and not two passes.** A debt forced at year *y* becomes
eligible again at *y + intervallo*. If that falls inside the plan, the
*normal* loop may cut it, adding to `year_total`, breaking earlier, and
starving some parcel the previous pass had reached. That parcel is a new debt.
On the current data every debt lands at 2035 or later, too late to return, so
one extra pass converges — but the loop is what makes the guarantee true
rather than merely true of this dataset.

**Termination.** Each iteration adds at least one parcel to `placements`, a
parcel is placed at most once, and the parcel set is finite. Iterations are
therefore bounded by the number of fustaia parcels; exceeding the bound raises
rather than looping.

## Why one code path

Parcels are dynamically independent: `growth_tables(data)` is built once at
`simulation.py:347` from the original tree data and never rebuilt, `year_step`
does per-tree lookups keyed by `(compresa, genere, classe)`, and mortality is
a uniform scalar. Harvesting *p* cannot change how *q* grows. The only
coupling between parcels is `year_total`.

This is what licenses re-running the whole simulation instead of simulating
each debt parcel forward in isolation. Both are equivalent; re-running reuses
the existing engine, so gap tracking, second cuts, growth and ageing all fall
out for free, with no parallel implementation to drift out of step.

## Interaction with existing parameters

| parameter | interaction |
|---|---|
| `intervallo`, `intervallo_anno` | unchanged; a forced harvest sets `last_harvest`, so a debt parcel rests normally afterwards |
| `particelle_min` | unchanged; part of the commit condition, not of eligibility |
| `prudenza` | unchanged; applies to forced harvests identically |
| `riduzione` | unchanged; applied downstream to all events |
| `ordine` | unchanged; the option adds harvests, it does not reorder |
| `volume_obiettivo` | still governs the normal loop; forced harvests may push a year above it, by design |
| `anno_eta` | unchanged |

`plan_events` memoizes on a parameter tuple; `prelievo_completo` joins it, so
`@@piano_di_taglio` and `@@prelievi` still share one simulation run.

## Output

The existing per-year diagnostic gains a line naming forced harvests, so the
recovery is visible on stderr during a build:

```
  @@piano_di_taglio anno 2036: recupero Fabrizia/11a (2677 m³, debito 2035)
```

No change to the rendered tables: forced harvests appear as ordinary rows.
`relazione.tex` needs a paragraph in `sec:note:piano` describing the
mechanism, since it documents the annual cycle step by step.

## Expected results on current data

Placement under the specified rule:

| debt | incurred | placed | harvest |
|---|---|---|---|
| Fabrizia/11a | 2035 | 2036 | 2,677 m³ |
| Capistrano/1 | 2035 | 2037 | 1,886 m³ |
| Capistrano/6a | 2035 | 2040 | 524 m³ |
| Capistrano/2 | 2036 | 2039 | 732 m³ |
| Capistrano/7a | 2039 | 2040 | 24 m³ |

Annual totals, thousands of m³:

```
              2027  2028  2029  2030  2031  2032  2033  2034  2035  2036  2037  2038  2039  2040   max   sd
base          14.3  13.6  14.0  13.9  18.3  13.0   5.3  24.3  18.6  12.7  12.7  14.4  13.5  13.1  24.3  4.0
completo      14.3  13.6  14.0  13.9  18.3  13.0   5.3  24.3  18.6  15.4  14.6  14.4  14.2  13.6  24.3  3.9
```

Plan total 201,844 → 207,687 m³ (+5,843, +2.9%). The peak year is unchanged:
2034 already carries 24.3k in the base plan, and the debts fill the 2036–37
valleys rather than creating a new maximum, so the profile becomes marginally
more even than the base plan's own.

## Testing

Regression:
- `prelievo_completo` absent or `no` reproduces current events exactly; golden
  files regenerate unchanged.

Unit, on synthetic forests:
- a parcel eligible but unreached becomes a debt;
- a resting parcel does not;
- a parcel whose rules yield zero does not (comparto F, and a below-floor
  fustaia);
- `debt_size` is read at the debt year, not the placement year;
- placement selects the lowest-total eligible year, ties to the earliest;
- drain order is total: permuting input parcel order leaves placements
  unchanged;
- stickiness holds across iterations;
- the iteration bound raises instead of looping.

Integration:
- forced harvests are absent from `year_total`: the committed set with the
  option on is a superset of the set with it off, and agrees on every
  non-forced parcel;
- **guarantee property** — with the option on, no parcel has an eligible year
  and zero harvests.

## Deferred

- `docs/prelievo-completo-it.md`, matching the `ceduo.md` / `ceduo-it.md`
  convention.
- Whether recovery harvests should be marked in the rendered tables rather
  than appearing as ordinary rows.
- The young comparto-A question described under Scope.
