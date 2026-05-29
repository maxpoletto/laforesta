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
  equazioni da CSV" goes away, and the plan-level "Esporta CSV" no longer
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
  never alter already-recorded measurements. They reach Ipso the next time it
  syncs (Part B).

- At installation, Abies attempts to import a regression CSV into the initial
  active set; a missing file is a warning, not a fatal error.

- The work is split into two stages:
  - **Part A** — Abies side: schema, settings section, compute/import/export,
    and repointing the mark-entry form. This document specifies Part A fully.
  - **Part B** — Ipso integration: the Abies→Ipso sync endpoint and Ipso-side
    polling/notification (sketched under "Synchronization with Ipso"). Part B is
    a prerequisite for the broader workflow in which Ipso mark uploads flow into
    Abies automatically.

## Data model

Three tables replace `tree_height_regression`, which is removed. Since there is
no production data, the change is folded into migration `0001_initial` rather
than added as a later migration.

- **hypso_param_set** — one row per parameter set.
  - `id`.
  - `source`: how the set was produced — `computed` (fitted from surveys) or
    `imported` (loaded from CSV).
  - `min_n`: the minimum per-group sample count used when computing (NULL for
    imported sets).
  - `created_at`: when the set became active — the start of its live interval.
  - `superseded_at` (nullable): when the set stopped being active. NULL marks
    the single currently-active set. The pair (`created_at`, `superseded_at`)
    is the historical "live from–to" interval.
  - Invariant: at most one row has `superseded_at IS NULL`. Enforced in the
    single write path (compute/import/clear all run in one transaction that
    closes out the current set before opening a new one).
  - Carries `version` and is tracked by django-simple-history.

- **hypso_param** — one row per (set, region, species).
  - `id`, `set_id` (FK → hypso_param_set, CASCADE), `region_id` (FK → region,
    PROTECT), `species_id` (FK → species, PROTECT).
  - `func`: regression family; currently always `ln` (`h = a·ln(D) + b`).
  - `a`, `b`: regression coefficients. `r2`: coefficient of determination.
  - `n`: number of samples used for this (region, species) fit.
  - `UNIQUE(set_id, region_id, species_id)`.
  - Immutable once written: a set is created whole and never edited row by row.
    Its audit trail is the retained set, not a per-row history.

- **hypso_param_set ↔ survey** — provenance, modeled as a many-to-many from
  `hypso_param_set` to `survey`: the surveys whose samples fed a computed set.
  Empty for imported sets. This records "this set is based on this data"; it is
  not the historical log.

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
  computed sets — the list of source surveys. If no set is active, it says so.

- A read-only **sortable table** of the active set's parameters. Columns:
  Region, Species, Function, a, b, n, r². Parameters are managed as a whole
  set, so there is no per-row add/edit.

- Toolbar buttons, above-right of the table:
  - **Esporta CSV** — downloads the active set as `equazioni_ipsometro.csv`
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
CSV backup first; "Esporta CSV" remains available for an explicit export.

## Effect on Piano di taglio

- "Modifica piano > Importa equazioni da CSV" is removed, along with the
  equazioni CSV in the plan-level export and the per-plan
  `tree_height_regressions` data.
- The "Nuovo albero martellato" form auto-populates `h = a·ln(D) + b` from the
  single active parameter set (keyed by region + species) instead of a per-plan
  table. The field stays editable; an override sets `h_measured = true`; a
  missing (region, species) entry leaves h blank for manual entry.

## Synchronization with Ipso (Part B)

Deferred to the second stage; sketch only.

- Abies exposes an endpoint that serves the active set as a CSV for Ipso.
  Because Ipso runs on a different origin and holds no Abies session, the
  endpoint is token-authenticated (reusing Ipso's existing bearer-token
  mechanism) and CORS-enabled — the first piece of the broader Ipso↔Abies
  channel that will also carry mark uploads.
- The downloaded data completely replaces Ipso's hypsometric data. Ipso's other
  reference data (species, terreni) is out of scope for this sync.
- Ipso checks for new data every 10 minutes while online, applies any update,
  and notifies the user with a dismissable modal. The service worker treats this
  request as network-only — it must not serve a cached copy.
