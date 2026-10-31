# M2: proposed edit-version and modification-time contract

30 October 2026. Reviewed revision: `8d0aaa98e30ec1d00de93f66f49262e73282a37d`.
Scope clarified on 31 October 2026 against `5c03af5` (after M1).
Implementation is in progress. The table records the pre-M2 behavior; completed
rows and verification are recorded below. Keep/refactor-only rows retain their
existing behavior unless a separate change is justified.

## Recommendation

Keep the existing `version` field and wire name. Define it as an **edit
concurrency token**, and make `modified_at` reflect changes to the row's stored
domain data, including materialized totals. Keep content ETags as the identity
of published digests. These three values have distinct responsibilities:

| Value | Question it answers | What it must not imply |
| --- | --- | --- |
| `version` | Is this edit based on an accepted editable state? | Every child row or cached aggregate is unchanged |
| `modified_at` | When was this row's stored domain state last written? | A monotonic concurrency token or an entire subtree's latest event |
| Digest ETag | Are the bytes of this published digest unchanged? | An individual row's edit permission/version |

A new tree changing an intervention's total should refresh that total and its
timestamp. It should not necessarily reject a still-valid intervention note
form. If the same operation advances intervention state, that state change
must invalidate the old form.

## Current behavior and evidence

The original 30 October source inventory found 21 concrete `TimestampedModel`
subclasses and 22 explicit `save(update_fields=...)` call sites across application
Python code, excluding migrations. Those counts predate M1 and are discovery
aids, not a count of defects or separate implementation tasks. The cases below
have been checked against the post-M1 code.

[`TimestampedModel.save()`](../apps/base/models.py) validates Decimal storage;
it does not increment `version`. `modified_at` is `auto_now=True`, but a partial
save that omits that field does not update it. Full saves and many views
increment versions manually. The main cases are:

**How to read the scope column:** **Fix** changes stored behavior; **Refactor**
changes code ownership while retaining the stated behavior; **Keep** requires
no behavior change. Some rows need both a fix and a refactor. These are paths
through shared code, so fixing the base partial-save rule addresses several
rows at once.

| Current path | Current version behavior | Current timestamp behavior / implication | Proposed behavior: what changes and what stays | Scope / practical effect |
| --- | --- | --- | --- | --- |
| [`TimestampedModel.save()`](../apps/base/models.py), the common base | Leaves version policy to its caller | A nonempty partial save updates `modified_at` only if explicitly listed; an empty field list saves nothing | Automatically include `modified_at` in nonempty partial saves. Keep caller-owned version increments and empty saves as no-ops. Full saves already update the timestamp. | **Fix.** One base-class change supplies the timestamp corrections listed below; no automatic version bump on every save. |
| [`save_model_response()`](../apps/base/responses.py) | Checks submitted version and increments on update | Full save updates `modified_at` | Same version check, one increment and timestamp update per accepted edit. If a lower-level edit helper is introduced, delegate the increment to it exactly once. | **Keep; optional refactor.** No new conflicts, fields or UI behavior. Existing correct behavior becomes an explicit contract. |
| Manual intervention edit and state transitions in [plan views](../apps/piano_di_taglio/views.py) | Parent version increments | Full save updates parent timestamp | Same version increment and timestamp update. Keep loading the current row under lock and accepting only editable fields. | **Keep; optional refactor.** A note/state edit remains an edit; submitted forms still cannot overwrite calculated totals. |
| First mark, [`auto_advance_to_marked()`](../apps/piano_di_taglio/mark_import.py) | Parent version increments when state changes | Full state save; a changed total then produces another save/history event | Combine state/date advancement and total into one parent save, with one version increment and timestamp update. Preserve the existing transition condition and date rule. | **Fix + refactor.** One history event contains the resulting state and total. A form opened before the state change still conflicts. |
| Later mark total, `rematerialize_volume_marked()` in the same module | No parent increment | Partial save of `volume_marked_m3` leaves timestamp unchanged; SUM is read before the parent lock | Keep parent version unchanged. Lock the parent before reading SUM, and update only the changed total plus timestamp. Keep unchanged totals as no-ops. | **Fix + refactor.** Timestamp reflects total changes; an older intervention note form remains valid and preserves the current total. The lock order avoids publishing a pre-lock aggregate. |
| First manual harvest on an open intervention, [harvest save](../apps/prelievi/views.py), `save_view()` | Parent version increments for open → harvesting | Full state save; a changed actual total then produces another save/history event | Combine the existing state advancement and changed total into one parent save, with one version increment and timestamp update. | **Fix + refactor.** Same business transition and form-conflict behavior; one coherent parent history event, as for the first mark. |
| Later harvest total, [`_rematerialize_volume_actual()`](../apps/prelievi/views.py) | No parent increment | Partial save leaves timestamp unchanged; SUM is read before the parent lock | Keep parent version unchanged. Read SUM after locking the parent; save the changed actual total and timestamp. Keep unchanged totals as no-ops. | **Fix + refactor.** Same total and form-validity policy; reliable timestamp and the same serialization boundary as marked totals. |
| [Harvest CSV apply](../apps/prelievi/csv_harvests.py), `apply()` | Does not increment the intervention version for totals | Reimplements actual-volume recomputation, including the timestamp omission and SUM-before-lock sequence | Call the same totals service as manual harvest writes. Keep total-only version behavior and the CSV path's existing state-transition policy; update timestamp only for changed totals. | **Fix + refactor.** Removes a second policy owner. This does not add automatic state advancement to CSV imports. |
| [Parcel metadata edit](../apps/bosco/views.py), `parcel_metadata_save_view()` | Version increments | Partial save omits `modified_at` | Keep the version increment; partial save also updates `modified_at`. | **Fix through the base rule.** Parcel metadata edits acquire an accurate row modification time. No new form field or version rule. |
| [Existing-parcel CSV update](../apps/base/csv_parcels.py), `update_existing()` | Version increments | Partial save omits `modified_at` | Keep the existing changed-row/version policy; include timestamp when a row is saved. | **Fix through the base rule.** Imported parcel corrections update their timestamps; unchanged rows remain skipped. |
| [Observation edit](../apps/bosco/views.py), `observation_save_view()` | Version increments | Partial save omits `modified_at`, even for text/location edits | Keep one version increment for the observation edit, including its form-owned photo/category changes; update the observation timestamp too. | **Fix through the base rule.** The observation's existing edit boundary is retained; no independent photo/category edit token. |
| [Hypsometric-set archive](../apps/base/hypsometry.py), `_archive_active_sets()` | Version increments | Partial save omits `modified_at`; `superseded_at` still records archival time | Keep the increment and explicit archival time; also update the archived set's `modified_at`. | **Fix through the base rule.** Corrects set metadata; height rows remain immutable and M1's archived equations/provenance are preserved. |
| [Reference CSV updates](../apps/base/csv_reference.py), `apply()` | Increments changed versioned rows | Explicitly includes `modified_at` | Same version and timestamp behavior. The explicit timestamp inclusion can remain or become redundant under the base rule. | **Keep.** Already follows the proposed contract; no importer rewrite is needed for the timestamp fix. |
| [Reserved PAI survey repair](../apps/base/preserved_trees.py), `get_preserved_tree_survey()` | Changes an incorrectly true `active` flag without incrementing | Partial save omits timestamp | When correcting `active`, increment the survey version once and update its timestamp. Leave an already-valid survey untouched. | **Fix.** The automatic metadata correction becomes visible to edit concurrency. This needs one explicit version increment in addition to the shared timestamp fix. |
| [`User.save()`](../apps/base/models.py), added in C14 | Database counter advances on ordinary saves; login-only/empty saves excluded | User deliberately has no `modified_at` | Keep C14 unchanged; exclude User from the generic domain edit helper. Do not add a timestamp. | **Keep.** No account schema, login or last-admin policy changes; avoid accidentally incrementing twice. |
| `SiteSettings` in the same module | No version | No timestamps; singleton writer lock instead | Keep its current locked, last-save-wins behavior and document it. Do not add fields or stale-form guards in this work. | **Keep.** A versioned settings form would be separate work. This exception concerns SiteSettings, not M1's Species/equation snapshot guard. |
| [`IpsoUpload.save()`](../apps/ipso/models.py) | No edit version | Explicitly adds `updated_at` to partial saves | Keep this staging model's existing status-clock contract; do not route it through TimestampedModel/edit-version helpers. | **Keep.** No staging, mobile-storage, session or upload-protocol change. |
| M1's [Species/equation editor](../apps/impostazioni/species_equations.py), `save_configuration()` | Checks Species version and equation snapshot; increments Species on an accepted popup edit | Full saves update species/changed volume-equation timestamps; height replacement archives the old set | Preserve the version and snapshot checks, atomic save and partial-update semantics. Only set archival gains the timestamp correction described above. | **Keep**, apart from that shared fix. No replacement of the new editor or additional equation token. |

For a concrete example, suppose an intervention is at version 7. Recording a
later tree that changes its total from 10 to 12 m³ would leave version 7 both
before and after M2. Today its timestamp stays old; after M2 the timestamp would
advance. A note form opened at version 7 can still save and must retain 12 m³.
By contrast, recording the first tree on a planned intervention also moves it
to marked: the parent becomes version 8, so the old note form must be rejected. The
proposed change combines that state change and total into one save rather than
changing which of those two situations constitutes an edit conflict.

### Bounded implementation scope

The initial work is the shared partial-save timestamp fix, consolidation of
intervention total/state writes (including locking before aggregate reads),
the reserved-PAI-survey correction, regression tests and accurate documentation.
The aggregate values, units, rounding and null-versus-zero rules stay the same.
These changes need no schema migration, timestamp backfill, new form token,
CSV/mobile format change or reorganization of every writer in the application.

A common edit-save helper is **optional follow-on refactoring**, not a
prerequisite for those fixes. The shared-Tree identity question below is a
**separate investigation**, not an assumed defect or a commitment to add
another token. User, SiteSettings and IpsoUpload are included in the table to
define exceptions, not to propose changing them.

The existing materializers avoid writing when their aggregate is unchanged.
Retain that behavior. Also preserve their different empty-set semantics:
marked volume uses the database SUM result (`None` for no known volumes),
whereas actual harvest volume becomes zero when the SUM is empty.

This is not evidence that current intervention edits overwrite newer totals:
`item_save_view()` reloads the current row under lock, and `_parse_item_body()`
does not accept `volume_marked_m3` or `volume_actual_m3`. Keep both protections.
The problem is an implicit contract and misleading timestamps, with the risk
that a later refactor assumes a stronger version guarantee than exists.

## Proposed write rules

### 1. Editable state

Creation starts at version 1. An accepted edit operation checks the current
submitted version under the write lock, then advances it once. Its changed
fields and `modified_at` are persisted in the same transaction. Keep the
current conservative behavior of accepted form saves advancing the token even
when submitted values happen to equal the current values; the token is not a
content hash. Existing batch importers can continue skipping unchanged rows.

“Editable” includes meaningful automatic decisions, such as an intervention
moving from planned to marked or from open to harvesting. It also includes
form-owned related data: opening-date corrections are guarded by the parent
intervention version, and observation photo/category edits belong to the
observation's edit operation.

An edit that recalculates its **own** derived fields is one edit. For example,
changing a tree's diameter writes its new diameter, volume and mass with one
TreeMark/TreeSample version increment. It is not split into an edit and a
second derived write.

### 2. Derived values on another row

Recomputing `HarvestPlanItem.volume_marked_m3` or `volume_actual_m3` updates
that row's `modified_at`, but leaves its edit version unchanged when no
editable intervention field changes.

Require a freshly loaded/locked parent and persist only the derived fields
plus timestamp. A service cannot accept arbitrary fields or a stale parent
instance. This is what makes it safe for an existing metadata form to remain
valid. It is also safe when the metadata edit commits first: the materializer
must not write metadata from an older in-memory object.

On the first mark/harvest, combine any state transition and changed total into
one parent save: editable fields changed, therefore advance the parent version
once and timestamp once. Later total-only updates leave the version alone.
An identical aggregate should cause no save, timestamp movement or history row.

### 3. Exceptional models and infrastructure

Do not automatically give every table every field:

- Keep C14's User counter and login-only exception. Do not increment it again
  in a generic domain helper. No new User timestamp is needed for M2.
- Keep `IpsoUpload.updated_at` as a staging/status clock, not a forestry edit
  token. Its state transitions and retry identity remain separate contracts.
- Keep immutable HypsoParam rows immutable. Archiving their containing set is
  a semantic set change and must also update its existing `modified_at`.
- Keep HarvestTransition guarded by the intervention edit; do not add an
  independent form version without a separately editable transition endpoint.
- Keep nonce receipts, DigestStatus counters, reference-only tables and join
  tables under their own contracts. Lock-only no-op writes must not create
  versions, modification times or audit entries.
- Initially retain SiteSettings' locked, last-save-wins behavior and document
  it. If simultaneous settings editors become a use case, add one explicit
  settings version with form guards in a separate change. A universal save
  override would not solve stale forms by itself.

For the PAI reserved-survey `active` correction, use a semantic metadata save
when a correction is needed: version and timestamp advance once. Leave a valid
row untouched. Keep bootstrap/repair entry points explicit.

## Implementation boundaries

### Make timestamps reliable in the base class

For existing TimestampedModel partial saves, add `modified_at` whenever a
nonempty `update_fields` set represents a real model write. Preserve empty
`update_fields` as a no-op. Keep Decimal validation and normal history signals.
Do **not** add automatic version increments to this base class: it cannot know
whether an update is a metadata edit or parent-total maintenance.

Prefer model saves over raw queryset updates for audited domain writes. A
necessary bulk/update path must explicitly set the timestamp, preserve the
version policy and document history handling. Infrastructure no-op lock writes
remain exempt. Do not use wall-clock comparisons as a replacement for versions.

### Use narrow operation helpers

1. An optional follow-on edit-save helper, for TimestampedModel objects already
   protected by a lock/version check, owns the one increment and explicit field save. It does
   not parse HTTP requests or automatically touch parents. Migrate callers in
   small groups; never leave both caller and helper incrementing.
2. A plan-item totals service owns marked/actual recomputation, changed-field
   detection, null-versus-zero rules and parent persistence. Both harvest CSV
   and manual harvest paths use it. Callers retain transaction/nonce ownership;
   the service returns the affected parent identity/current state for patches.
3. The first-mark/first-harvest operations decide state changes before that
   parent persistence step, so they do not save the same logical parent edit
   twice. Preserve the current source of `date_actual`; changing its business
   meaning is outside M2.

Choose one lock order across these callers: determine affected parents, lock
them in ID order, then mutate children and compute the totals under that
boundary. An aggregate read before a contended parent lock can become stale.
Revalidate PostgreSQL foreign-key/row-lock interactions as part of the backend
move; merely extracting today's query sequence does not prove serialization.

Use explicit allowlists for edit fields and derived fields. Do not introduce a
generic `save(..., bump_version=False)` option on all models: that makes policy
easy to bypass and conceals the reason for each write.

Keep dirty marks and response construction within the existing C1/C2/C11 nonce
transaction. Rebuild side-effect patches from the post-write row. ETags already
cover changed digest content; no second digest revision field or timestamp
validator is needed.

## Parent and shared-identity boundaries

A parent's token does not claim that all children are unchanged. Continue
checking live invariants under locks when narrowing an intervention target,
transitioning state or deleting a protected aggregate. Derived totals alone
must not substitute for existence/scope checks on actual children.

There is a further boundary worth testing separately: sample/PAI edits check a
TreeSample version but also change its shared Tree identity. Other visits to
that same Tree have different TreeSample versions. Incrementing Tree.version
does not protect a form unless the form checks it. Before declaring identity
edits fully protected across visits, add a two-visit competing-edit regression.
If it confirms an overwrite, require both observation and Tree identity tokens
when changing shared fields; do not bump every historical visit indiscriminately.
Keep that extension separate from the initial timestamp/materialization change.

`modified_at` describes a row, not recursive activity. For example,
[`build_grid_record()`](../apps/base/digests.py) currently derives its displayed
last-update value from the grid and its sample areas, not all survey/tree
activity. Document that scope rather than silently broadening it or touching
every ancestor for every tree insertion.

## History, rollout and migration

Continue recording changed aggregate values through normal history saves;
the audit field mapping already includes intervention totals. Combine parent
writes where possible, but do not rewrite old history to remove its previous
two-step events. Timestamp/version-only history is not itself a new visible
field change in Controllo.

The initial proposal needs no database schema migration: retain all existing
fields and values. Old timestamps cannot in general be reconstructed reliably,
especially for deliberately untracked Tree/Sample rows. Do not backdate them
from unrelated children or rewrite historic versions. Subsequent writes acquire
the clarified behavior; document that pre-upgrade timestamps have the old limits.

Update CLAUDE.md's universal-save claim, the base model docstring,
`docs/database.md`, `docs/data-architecture.md`, and affected page docs when the
behavior is implemented. This proposal does not change those contracts yet.

## Suggested sequence and acceptance tests

Each step starts with failing tests and is independently reviewable:

1. **Partial-save timestamps.** Test parcel edits, parcel CSV updates,
   observation edits, hypso-set archival and parent totals. Freeze/control time
   so tests prove movement and unchanged `created_at`. Empty saves stay no-ops;
   normal save/history/Decimal behavior remains intact.
2. **One totals owner.** Test manual, CSV and Ipso mark/import paths plus
   harvest create/edit/delete/CSV. Assert exact totals, `None` versus zero,
   parent version behavior, one-save transitions, unchanged-total no-ops and
   matching response/digest rows.
3. **Reserved PAI survey correction.** Cover the missing semantic version
   increment and unchanged-row no-op. Keep User, SiteSettings and staging
   exceptions explicit. This completes the initial behavior-fix scope.
4. **Optional edit-helper cleanup.** If useful after those fixes, introduce the
   narrow edit helper and migrate explicit increments in small domain groups.
   This should preserve already-correct behavior, not widen version semantics.
5. **Separate shared identity investigation.** Reproduce the two-visit Tree
   boundary before proposing a new wire token. Add the smallest guard supported
   by that test.

Critical interleavings for step 2:

- A later tree changes only the total; an older intervention note form still
  saves and preserves the new total.
- First-tree state advancement makes that old form conflict.
- Metadata edit then total update, and the reverse order, preserve both changes.
- Repeated materialization is a no-op; concurrent child writers do not publish
  an old aggregate. Read the aggregate under the serialization boundary, not
  before acquiring the parent lock.
- A rejected edit, failed history save, dirty-mark failure or nonce-receipt
  failure rolls back timestamps, versions, totals and history together.
- A derived-only change alters the digest ETag even when edit version stays
  unchanged. C6/C7 behavior must remain intact.

Use real independent connections for overlapping-write tests and run the same
service contract on PostgreSQL when that backend is introduced. SQLite-specific
tuning remains lower priority; these write rules are backend-independent.

## Implementation progress

Each numbered row refers to its position in the current-behavior table. Rows
fixed by the common base rule receive their own endpoint/service regression
commit; they do not repeat the timestamp implementation.

| Row | Result | Verification |
| --- | --- | --- |
| 01 — TimestampedModel | Implemented; normalize partial fields once and advance only the row clock | Two regressions failed before the fix; timestamp/model suite: 64 passed |
| 04 — First mark | One state/date/total save for manual and imported marks; shared parent lock | Both new history regressions failed first; marking suite: 208 passed |
| 05 — Later marked totals | Shared locked-sum service; parent-first deletion; unchanged total is a no-op | Pre-lock interleaving regression failed first; totals/rollback tests: 38 passed |
| 06 — First manual harvest | One state/actual-total event; revalidate the worksite after locking it | New history regression failed first; create tests: 6 passed |
| 07 — Later actual totals | Shared locked-sum service; parent-first edit/delete; zero and no-op rules preserved | Pre-lock sum regression failed first; harvest/totals/rollback suite: 111 passed |
