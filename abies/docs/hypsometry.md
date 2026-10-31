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
recompute existing rows. The downstream configured volume/mass calculation has
two implementations: JS (`volume.js`, interactive forms) and Python
(`equations.py` using the numeric core in `tabacchi.py`, all server entry paths).
They agree within 1 ULP (≤0.0001 m³ / ≤0.001 q)
but cannot match exactly (JS float vs Python Decimal); the bound is locked by
`test/test_tabacchi.py`.

The parameters a and b are obtained by regression over a set of sampled trees,
per (region, species) pair. If a (region, species) pair lacks parameters, that
is fine: a marked-tree height can be measured or entered manually, or stored as
unknown (`h_m=NULL`, `h_measured=false`). An unknown height leaves its
Tabacchi volume and mass unknown too.

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

## Ownership and lifecycle

Height equations are global, independent of harvest plans. At most one
parameter snapshot is active. Superseded snapshots are archived with their
live interval; a UI to browse the archive is deferred. Plan-level edit/export
has no equation import/export or per-plan regression data.

Writers and admins manage height and volume equations in the Species section
of Impostazioni. The species summary shows height coverage by compresa and
volume availability. Its popup edits species parameters, one volume equation
or explicit reference, and regional height equations together. See
[`page-impostazioni.md`](page-impostazioni.md#species-equation-editor).

A change takes effect for future calculations. It never rewrites captured
heights, volumes, masses or intervention totals. A species rename changes its
labels while retaining every equation association. Ipso receives current height
parameters when it refreshes its reference bundle.

At installation, Abies attempts to import a regression CSV into the initial
active set; a missing file is a warning, not a fatal error.

## Data model

Equation identity is independent of editable common names. Height rows link to
Species and Region by ID; desktop digests include hidden `species_id` and
`region_id` columns for automatic-height lookup. A species rename refreshes the
height digest and labels without changing its regression associations.

Volume coefficients now live in species-owned `VolumeEquation` rows. Desktop
previews receive the current coefficients on species options; server calculations
resolve the same species ID. The old name-keyed tables are initialization and
legacy parity fixtures, not the runtime source for tree calculations. An absent
equation gives an unavailable estimate; the entry form explains the missing
volume equation instead of showing zero. Negative estimates still clamp to zero.

This does not yet localize the species reference catalogue. Some Italian common
names remain in default initialization, legacy name-map compatibility and
numeric parity fixtures. Frozen migrations retain their original labels so
upgrades are reproducible. Current Species names are editable stored data, not
automatically translated labels; changing a string catalogue alone does not
translate those records. A future catalogue cleanup can identify real taxa by
Latin name, localize their default common names and preserve user overrides.
Catch-all categories need their own identity because they are not individual
taxa. Neither change is required for runtime equation associations to survive
renaming: those already use IDs.

`merge_active_set()` publishes an archived snapshot while preserving every
species/region pair outside the change. Unchanged rows retain source and original
set provenance. Imported/computed sets remain supported; mixed/manual edits can
omit `n` and `r2` when there is no measured fit behind an equation.

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
- Inputs are the `(d_cm, h_m)` pairs of eligible `tree_sample` rows. Eligible
  rows have `h_measured = true`; **coppice tree rows are excluded** because
  hypsometry serves high-forest marking only; and rows with non-positive or
  missing D or h are dropped.
- Rows are grouped by (region, species): region via
  `tree_sample → parcel → region`, species via the sampled `tree`. This
  allows both structured and unstructured surveys to participate. A group is fit
  only if it has at least `min_n` eligible rows; smaller groups produce no
  entry.

## Height tools inside Species

The species popup has one row per compresa. Enabling a row supplies `a`, `b`
and optional N/r²; disabling it removes that pair only. A save publishes a new
snapshot, preserves other species and regions, and retains the original source
of unchanged rows. Metadata and all equations save atomically. Invalid values,
proxy cycles, removal of an equation still needed by a proxy, and stale forms
are rejected without partial writes.

The Species toolbar offers separate height and volume CSV imports/exports.
The reviewed import flow below updates only supplied equations. There is no
standalone read-only height table or global delete button.

The **Calcola equazioni di altezza** panel shows the active snapshot's source
(computed/imported/manual/mixed), activation date, minimum N, source surveys,
and whether those surveys drive Bosco's height plots. Its controls are:

- Minimum N: the per-group minimum sample count.
- Surveys: structured or unstructured surveys with measured non-coppice trees;
  each option displays that eligible-row count.
- Use these surveys for height/diameter charts: accepted computations can also
  supply the survey list for Bosco's height scatter plot. Other dendrometry
  uses the separate Parametri dendrometrici setting. A later partial edit retains
  this chart setting; the original per-row sources remain independently visible.

### Compute → accept/reject flow

1. Select minimum N and surveys, then click Calcola.
2. Abies computes an in-memory candidate without writing the database. Eligible
   rows are grouped by region/species IDs and fitted as described above.
3. Review the candidate table with coefficients, N and r². Reject discards it.
4. Accept recomputes the candidate server-side and verifies both its values and
   the configuration against the preview's fingerprints. Changed inputs or
   configuration require a refreshed preview. The accepted pairs replace only
   their own entries in a new snapshot; all omitted pairs remain intact.

Archives preserve the old snapshot on every change. Unchanged copied rows
retain their original source and origin-set reference, including the source
surveys when computed. Clearing the active set remains an explicit archival
service/API operation for existing callers; it is not exposed as an ordinary
settings action or implied by a partial import.

## Reviewed batch imports

`POST /api/impostazioni/equations/import/preview/` accepts `kind=height` or
`kind=volume` and the base64 CSV `file`. It performs no writes. The response
contains the incoming/matched species, region, current/proposed values, errors,
and server-rendered mapping controls. Unknown or case-ambiguous names require
explicit `species_map`, `region_map`, or `proxy_map` entries keyed by CSV row
number. A missing mapping never creates a species or guesses an association.

Only an error-free preview has a confirmation token. It is signed, bound to
the authenticated user and configuration snapshot, and expires after 15 minutes.
`POST .../confirm/` accepts that token and a normal retry nonce. It rechecks
the configuration under the equation-write lock before applying all accepted
rows together with history, digest invalidation and the nonce receipt.
Changed configuration returns a conflict and requires a new preview; canceling
the preview writes nothing. Invalid/duplicate rows cannot be partially imported.

Both import kinds **upsert only supplied equations**, preserving every omitted
species/region pair. The same preservation rule applies to the existing height
import endpoint and computed parameter acceptance. Computation additionally
returns configuration/candidate fingerprints so the UI can detect a changed
preview before acceptance. Explicit clear remains an archival service/API
operation, not an implicit consequence of importing a partial file.

Readable height CSV (N/r² may be blank for manually supplied equations):

```csv
compresa,genere,funzione,a,b,n,r2
Capistrano,Abete bianco,ln,7.0306,-4.2563,48,0.6312
```

Volume CSV uses species names and either coefficients or an explicit proxy:

```csv
genere,b0,b1,b2,fonte,riferimento
Abete bianco,-1.8381,0.037836,0.39934,Tabacchi,
Altra conifera,,,,Tabacchi,Abete bianco
```

Exports use the installation's CSV delimiter/decimal convention; imports accept
both supported conventions through the shared CSV codec. Volume export is
`GET /api/impostazioni/equations/volume/export/`. Proxy rows round-trip without
copying coefficients or changing their species identity.

Historical tree repair is outside these endpoints. Equation edits/imports never
rewrite captured heights, volume, mass or intervention totals. A future repair
of previously unavailable Abete rosso estimates will be a separate one-time CLI
operation, with no production repair UI.

## Data serving

The active set is served as the `hypso_params` JSON digest
(`/api/impostazioni/hypso-params/data/`, readable by any authenticated role):
one row per active `hypso_param`, columns `row_id, Compresa, Specie, funzione,
a, b, n, r², region_id, species_id`. The final two columns are hidden internal
IDs. The Piano-di-taglio mark form joins by these IDs, independently of editable
labels. Computation previews use the same row builder. The description panel's
active-set metadata (source, date, min_n, source surveys, and whether those
surveys drive Bosco height plots) is a small separate JSON response, not a
cached digest.

Write → invalidation: compute-accept, import, and clear each mark
`species`, `hypso_params`, `parcel_dendrometry_points`, and `audit` stale. The
point digest is included because accepting/clearing/importing may switch Bosco's
Altezze chart between hypsometry-source surveys and the dendrometry setting.
`TestDigestInvalidation` in
`test/test_hypso_views.py` locks the contract: each write path re-reads the
served digest and asserts the change (import, for instance, asserts the
coefficients flowed through).

`replace_active_set()`, `merge_active_set()` and `clear_active_set()` own dirty
marks inside the same transaction as archiving/creating the parameter set. This applies to
direct service calls and requests without a nonce as well as UI requests.
A failure during dirty marking rolls back the parameter change; the endpoint
does not perform a separate post-commit invalidation.

## Effect on Piano di taglio

- Plan edit/export do not include per-plan hypsometric equation import/export
  or per-plan `tree_height_regressions` data.
- The "Nuovo albero martellato" form auto-populates `h = a·ln(D) + b` from the
  single active parameter set (keyed by region ID + species ID). The field stays
  editable; an override sets `h_measured = true`; a
  missing (region, species) entry clears an obsolete automatic height and shows
  that automatic height is unavailable. Measured heights remain editable. A
  missing volume equation is reported explicitly even before height is entered.
  A mark can be saved without height; Abies stores null height, volume and mass, retaining the diameter
  for tree-count and basal-area summaries.

## Synchronization with Ipso

Ipso receives the active hypsometric parameter set as part of
`/ipso/reference.json`, the same bearer-protected reference bundle that carries
species, parcels, sampling context, PAI context, and work-package options. The
protected response is served with `Cache-Control: no-store`, and the service
worker bypasses storage for `no-store` responses so a stale active set is not
replayed from the HTTP or PWA cache.

Ipso 1.5.5 receives an additive `hypsometry_by_id` map keyed by region ID and
species ID, alongside the legacy name-keyed `ipsometrica` map. New clients prefer
the ID map; if that map is present, a missing pair means unavailable and does
not fall back to a possibly misleading name match. Older cached bundles still
work through the legacy lookup. No IndexedDB key, session or upload-schema
migration is needed.

A refreshed reference bundle completely replaces Ipso's local hypsometric
parameters for future field entries. Existing local sessions keep their recorded
measurements; Abies never recomputes already-recorded tree heights after an
active-set change.

## Nonnegative harvestable volume

Abies applies the same policy in the Python Tabacchi calculation and its
JavaScript live preview: a negative equation estimate is zero harvestable
volume, and its derived mass is zero. This covers very small measured trees
below the fitted equation's intercept, without rejecting their measurements.
Positive estimates, however small, retain their normal stored precision; there
is no blanket minimum-diameter cutoff. Unknown heights/species remain unknown
estimates, not zero. This policy does not change the recorded diameter or height.
It changes new calculations/previews; previously stored estimates are not
rewritten by this calculation change.
