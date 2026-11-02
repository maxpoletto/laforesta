# Modelling discussion — harvest scheduling, July–August 2026

Working notes behind the changes to `pdg/simulation.py` and `pdg/harvest_rules.py`
and the corresponding text in `template/tex/relazione.tex`.

**Deliberately not checked in.** Figures are current as of commit `444e50a`;
several differ from numbers quoted earlier in the discussion, because the code
changed underneath them (see [Numbers that moved](#numbers-that-moved)).

The report text is already committed — `def1566` (basal-area-only rule for young
stands) and `f875d90` (§Prudenzialità della simulazione, which carries the
selection-order comparison below).

All measurements: 2027–2040, `intervallo=10` (9 for 2027–2031),
`volume_obiettivo=12000`, `mortalita=1`, `prudenza=80`, `particelle_min=3`,
`anno_eta=2026`, unless stated otherwise.

---

## Starting point

The plan as it stood at the start of this work: **59 interventions, 201 844 m³,
46 of 61 fustaia parcels**. Fifteen fustaia parcels never appeared. Thirteen
parcels received a *second* cut within the fifteen years while those fifteen
received none, and the split by compresa was lopsided — Serra 29 parcels,
Fabrizia 12, Capistrano 5.

Investigating why produced two independent findings.

## Finding A — the volume floor vetoed young stands

`max_harvest` applied the volume rule at every age. For a stand below 120% of
its comparto's *provvigione minima* the rule yields `pp_max = 0`, hence
`V_max = 0`, which blocks any cut however much basal area the stand carries.

Seven parcels are under 60 years old at some point in the plan, and **every one
of them is below its floor**:

| particella | comparto | ha | età 2026 | prov. m³/ha | soglia |
|---|---|---|---|---|---|
| Capistrano 12b | A | 4.3 | 33 | 192 | 420 |
| Capistrano 13 | B | 13.0 | 53 | 243 | 420 |
| Capistrano 14a | C | 6.0 | 33 | 0 | 300 |
| Capistrano 15 | C | 21.2 | 43 | 244 | 300 |
| Capistrano 16b | A | 7.0 | 33 | 216 | 420 |
| Capistrano 17b | A | 3.6 | 32 | 291 | 420 |
| Fabrizia 11c | A | 10.8 | 38 | 333 | 420 |

Traced to commit `9160b24` ("basal area and volume are not mutually exclusive
criteria"), which changed the young-stand branch from `math.inf` to `vol_max`.
Correct in principle — both criteria should apply — but it also imported the
volume *floor*, a mature-stand protection that a perticaia cannot by definition
satisfy.

Resolved with the `--limiti-giovani` flag rather than by reverting.

### The two rule sets

`--limiti-giovani {area,area_volume}`, default **`area`**. Only affects stands
under 60; stands at 60+ are governed by the volume rules either way.

```
prelievo_completo=no
  area_volume     57 interventi   198,537 m³   44 particelle
  area            62 interventi   200,186 m³   49 particelle
```

The five admitted by `area`: Capistrano 12b, 15, 16b, 17b, Fabrizia 11c.
(Capistrano 14a has no mature trees at all; Capistrano 13 needs
`prelievo_completo` as well — see below.)

A full row-level diff showed **only added rows** — no existing parcel changed
year or volume. Two reasons: all five land in 2033, the one year that misses its
target, so nothing is displaced; and the flag's other possible effect, lifting
the volume cap on a young stand *above* its floor, is empty here because no such
stand exists.

### Which limit actually binds

Measured across 167 committed harvests in the plan:

```
minimum age at harvest                : 65
harvests with a finite basal-area cap : 0 of 167
volume cap consumed: min 76%, median 97%, max 100%
```

Stands at 60+ receive `math.inf` for basal area — they have **no** basal-area
cap, so volume always binds and is consumed nearly to the limit. The flag
therefore cannot touch any of them. Basal area is the operative limit only for
young stands, because it was never operative anywhere else.

## Finding B — eligible parcels were never reached

Separate from the rules: parcels the rules permitted to cut, that the scheduler
never got to. Priority is descending mature volume/ha and the year loop stops
once `volume_obiettivo` is met. Because `intervallo` (10 yr) is shorter than the
plan (14 simulated years), the dense parcels come back off their rest period and
retake the quota before the queue ever drains — a closed cycle, not a delay.

Addressed with `prelievo_completo` (default off), specified in
`docs/prelievo-completo.md`. It collects parcels that were eligible in some year
and cut in none, places each in the eligible year carrying the least volume, and
re-runs with those harvests forced, iterating to a fixpoint.

```
limiti-giovani=area
  prelievo_completo=no    62 interventi   200,186 m³   49 particelle
  prelievo_completo=si    70 interventi   210,044 m³   57 particelle
```

Recovered: Capistrano 1, 2, 6a, 7a, 13, Fabrizia 4, 7a, 11a.

Capistrano 13 is the instructive case — it turns 60 in 2033, so it is eligible
only 2027–2032, and in each of those years the target was met before it came up.

## Finding C — the harvest loop stopped instead of skipping

`harvest_parcel` broke out of the selection loop at the first tree that would
exceed a limit. Thinning from below that is harmless — every later tree is
larger still — but it silently encodes the ascending order. Under any other
order a single oversized stem aborted the whole parcel-year.

Found while testing the selection-order sensitivity, where it invalidated the
first result. Changed to `continue` (`444e50a`), raising the plan 0.1%
(209 743 → 210 044 m³) and requiring two golden tables to be regenerated.

---

## Sensitivity analyses

All against the current plan (`limiti-giovani=area`, `prelievo_completo=si`),
**70 interventions, 210 044 m³**, matching `csv/piano.csv`.

### Selection order — dal basso vs dall'alto

```
                ripresa      alberi   provvigione 2041
dal basso       210,044     162,161            776,748
dall'alto       218,824      35,516            795,570
                 +4.18%                          +2.42%
```

Thinning from above harvests **more** and leaves **more** standing — strict
dominance, so from-below is the conservative assumption on both axes, not a
reallocation of volume between the harvest and standing columns.

The mechanism, from 5 608 measured mature trees:

```
classe     n   pv medio   incr. assoluto
    20   433     5.25%      0.0153 m³/anno
    30   749     3.56%      0.0283
    50   551     2.34%      0.0641
    70   109     1.73%      0.1068
    90    16     1.33%      0.1505
```

Percent increment falls with diameter; absolute current increment rises with it
by an order of magnitude. Thinning from below removes the trees of greatest
**percentage** increment and least absolute increment — so the report must say
*maggiore accrescimento percentuale*; the bare word would be read as the
absolute quantity and would assert the opposite of the truth.

### Prudence coefficient

```
c =  80%    70 interventi   210,044 m³
c = 100%    69 interventi   258,614 m³        +23.1%
```

A reader reproducing the arithmetic from the documented method without the
coefficient lands 48 570 m³ high. This is why `c` was added to §Parametri della
simulazione — stated neutrally, with no rationale given.

---

## Numbers that moved

Figures quoted earlier in the discussion that the code has since superseded:

| quantity | as quoted | now | why |
|---|---|---|---|
| plan total | 201 844 m³ | 210 044 m³ | rule set, `prelievo_completo`, skip fix |
| dall'alto | 205 974 m³ (−1.8%) | 218 824 m³ (+4.2%) | break→skip; 24 aborted parcel-years removed |
| recovered by `prelievo_completo` | 6 parcels | 8 parcels | skip fix meets the target with fewer parcels, starving two more |
| provvigione finale, dall'alto | +19 000 m³ | +18 822 m³ (+2.42%) | restated as a percentage |

The **−1.8%** figure was wrong and its sign was wrong. It measured the
break-not-skip artefact rather than the selection order, and would have led the
report to claim the opposite of what the model does.

## Open items

- **`art. 49`** — §note-ripresa cites `artt. 48 e 49` jointly for both limits,
  while the §ripresa paragraph added in `def1566` pins the basal-area rule on
  art. 49 specifically. Not verified against `calabria:2011`; it is the kind of
  citation a reviewing authority checks.
- **The PDF has never been built.** `pdflatex` is unavailable in the working
  environment, so all text changes are verified only through template expansion
  and static checks on `\ref` targets. Run `make pdf` before submission.
- **Hardcoded figures in the prose** — the six-parcel list in §note-ripresa and
  the 4% / 2,4% comparisons in §Prudenzialità della simulazione will drift
  silently if the survey data changes. They are correct as of `444e50a`.
- **`N. Alberi` column** — 162 161 stems from below versus 35 516 from above for
  the same plan. The column is far more sensitive to the modelling assumption
  than the volumes are, and is not meaningful where a parcel's prescription is
  not a thinning from below.
