# Hypsometry

## Background

The height (h, in m) of a tree can be estimated from its diameter (D, in cm)
using an equation of the form
```
h = a × f(D) + b
```
where f is usually the natural logarithm, ln.

This function estimates the height of marked trees (in the Abies marked-tree
data-entry form, or via the Ipso mobile app during timber cruising), which in
turn feeds the Tabacchi equations to estimate harvest volume and mass.

The estimated `h` is captured at write time: later parameter changes never
recompute existing rows. The downstream Tabacchi volume/mass has two
implementations — JS (`volume.js`, the interactive forms) and Python
(`tabacchi.py`, CSV import) — which agree within 1 ULP (≤0.0001 m³ / ≤0.001 q)
but cannot match exactly (JS float vs Python Decimal); the bound is locked by
`test/test_tabacchi.py`.

The parameters a and b are obtained by regression over a set of sampled trees,
per (region, species) pair. If a (region, species) pair lacks parameters, that
is fine: heights are then measured or estimated manually.

Historically the parameters were computed from samples taken while creating a
harvest plan (pdg-2026) and stored in a "regressions" CSV file like this:
```
compresa,genere,funzione,a,b,r2,n
Capistrano,Abete,ln,7.0306,-4.2563,0.6312,48
```
(n is the number of samples used to determine a and b; r2 is the coefficient of
determination.)

But there is no reason for samples or regression parameters to be tied to a
harvest plan. As more samples are collected, Abies should be able to recompute
better parameters from more data, independently of any plan.

## Proposal

- Decouple regression parameters from harvest plans. "Modifica piano > Importa
  equazioni da CSV" goes away, and the plan-level "Esporta" no longer
  includes `equazioni_ipsometro.csv`.

- At any time, at most **one** parameter set is active, in both Abies and Ipso.

- Parameters live in a dedicated, plan-independent store, managed from a new
  "Parametri ipsometrici" foldable section on Impostazioni (below "Specie"),
  visible to writers and admins.

- Superseded sets are **archived, not deleted**: each set records the interval
  during which it was live, forming a historical log of "which parameters were
  active from when to when". A UI to browse that log is deferred; the data is
  retained from the start.

- New parameters take effect immediately in Abies for future D→h mappings; they
  never alter already-recorded measurements. They are included in Ipso's
  bearer-protected reference bundle the next time the PWA refreshes
  `/ipso/reference.json`.

- At installation, Abies attempts to import a regression CSV into the initial
  active set; a missing file is a warning, not a fatal error.

- The implementation has two surfaces: Abies owns the schema, settings
  section, compute/import/export flow, and mark-entry autofill; Ipso receives
  the active parameter set through the same reference bundle it uses for
  parcels, species, sampling context, and work packages.

## Data model

Schema lives in [`database.md`](database.md) → "Hypsometric parameters":
`hypso_param_set` (one set + its `created_at`/`superseded_at` live interval and
the at-most-one-active invariant), `hypso_param` (one immutable row per
region+species), and the `params_surveys` provenance M2M.

A missing (region, species) entry is never an error: the mark-entry form simply
leaves h blank for manual entry.

## Regression computation

- The fit reuses pdg-2026's logarithmic regression (`LogarithmicRegression` in
  `pdg-2026/pdg/computation.py`): a least-squares fit of h on ln(D) yielding
  `a`, `b`, plus `r2` and the sample count `n`. The small numeric core is ported
  into Abies; numpy is added to Abies's dependencies for it.
- Inputs are the `(d_cm, h_m)` pairs of eligible `tree_sample` rows. **Coppice
  samples are excluded** — hypsometry serves high-forest marking only — and rows
  with non-positive or missing D or h are dropped.
- Samples are grouped by (region, species): region via
  `tree_sample → sample → sample_area → parcel → region`, species via the
  sampled `tree`. A group is fit only if it has at least `min_n` eligible
  samples; smaller groups produce no entry.

## Parametri ipsometrici settings section

Visible to writers and admins, below "Specie".

- A **description** above the table summarises the active set: its source
  (computed/imported), the date it became active, the minimum-N used, and — for
  computed sets — the list of source surveys. If the set also drives Bosco's
  Altezze chart, the description says so. If no set is active, it says so.

- A read-only **sortable table** of the active set's parameters. Columns:
  Region, Species, Function, a, b, n, r². Parameters are managed as a whole
  set, so there is no per-row add/edit.

- Toolbar buttons, above-right of the table:
  - **Esporta** — downloads the active set as `equazioni_ipsometro.csv`
    (Italian locale). Always available; non-destructive.
  - **Importa CSV** — uploads an `equazioni_ipsometro.csv`. After a confirmation
    that it replaces the active parameters (the current set is archived), the
    uploaded rows become the new active set (`source = imported`, no surveys).
  - **Elimina** — after a confirmation, archives the active set so that no
    parameters are active. Non-destructive: the set is retained in the log.

- A **"Calcola nuovi parametri"** sub-section with:
  - "N minimo" — integer input (the per-group minimum sample count).
  - "Rilevamenti" — a multiselect of all surveys (reuses the existing `surveys`
    digest).
  - "Usa questi rilevamenti per i grafici altezza/diametro" — when checked, the
    accepted computed set also supplies the survey list for Bosco's
    diameter/height scatter plot. (Other Bosco dendrometry use the Parametri
    dendrometrici survey setting.) Imported CSV sets never enable this.
  - a "Calcola" button.

### Compute → accept/reject flow

1. The user sets "N minimo", selects one or more surveys, and clicks "Calcola".
2. Abies shows an in-progress modal while it computes a **candidate** set in
   memory (no database write): it gathers eligible samples from the chosen
   surveys, groups them by (region, species), and fits each group with at least
   N samples.
3. The candidate is shown as a table in the modal, with "Accetta" / "Rifiuta".
4. **Accetta** persists the candidate as the new active set — recomputed
   server-side from the same inputs, so the stored set is authoritative rather
   than client-supplied — and archives the previously-active set. **Rifiuta**
   discards the candidate with no database change.

Because superseded sets are archived rather than deleted, replacing parameters
(by compute, import, or clear) is non-destructive. None of these actions force a
CSV backup first; "Esporta" remains available for an explicit export.

## Data serving

The active set is served as the `hypso_params` JSON digest
(`/api/impostazioni/hypso-params/data/`, readable by any authenticated role):
one row per active `hypso_param`, columns `row_id, Compresa, Specie, funzione,
a, b, n, r²`. It is consumed by the settings table and by the Piano-di-taglio
mark form's h auto-fill (keyed on region + species). The description panel's
active-set metadata (source, date, min_n, source surveys, and whether those
surveys drive Bosco height plots) is a small separate JSON response, not a
cached digest.

Write → invalidation: compute-accept, import, and clear each mark
`hypso_params`, `parcel_dendrometry_points`, and `audit` stale. The point
digest is included because accepting/clearing/importing may switch Bosco's
Altezze chart between hypsometry-source surveys and the dendrometry setting.
`TestDigestInvalidation` in
`test/test_hypso_views.py` locks the contract: each write path re-reads the
served digest and asserts the change (import, for instance, asserts the
coefficients flowed through).

## Effect on Piano di taglio

- Plan edit/export do not include per-plan hypsometric equation import/export
  or per-plan `tree_height_regressions` data.
- The "Nuovo albero martellato" form auto-populates `h = a·ln(D) + b` from the
  single active parameter set (keyed by region + species) instead of a per-plan
  table. The field stays editable; an override sets `h_measured = true`; a
  missing (region, species) entry leaves h blank for manual entry.

## Synchronization with Ipso

Ipso receives the active hypsometric parameter set as part of
`/ipso/reference.json`, the same bearer-protected reference bundle that carries
species, parcels, sampling context, PAI context, and work-package options. The
protected response is served with `Cache-Control: no-store`, and the service
worker bypasses storage for `no-store` responses so a stale active set is not
replayed from the HTTP or PWA cache.

A refreshed reference bundle completely replaces Ipso's local hypsometric
parameters for future field entries. Existing local sessions keep their recorded
measurements; Abies never recomputes already-recorded tree heights after an
active-set change.
