# Abies code review and improvement plan

Review date: 14 October 2026. Reviewed revision: `c3ddd6419c8e42802bf610b5dea4be604d4b150d`.

Progress updated: 28 October 2026. **C1–C11, C13–C14, S1–S4 and R1 are
fixed in this branch.** The status table and notes below describe the completed
work; the original source references, reproductions, and diagnosis remain a
record of the reviewed revision, not a claim that those defects are still
present. Completion here means implemented and tested; deployment is separate.

Abies has a sound small-application foundation: explicit permissions, substantial tests, server-authoritative calculations, historical records, and a deliberately simple deployment. The most important problems are gaps between individually reasonable components: a database commit and its retry receipt, an invalidated digest and its HTTP validator, a database backup and its photographs, or a successful IndexedDB write and its UI acknowledgement. These gaps can duplicate field records, preserve incorrect summaries, or make recovery incomplete.

The remaining data-integrity and security work should precede broad refactoring.
A move to PostgreSQL is now probable for PostGIS support. SQLite-specific
hardening and tuning therefore have lower priority; backend-independent
correctness, authentication, and database/media recovery requirements remain.

## Scope and evidence

The review covered models and migrations, request validation and permissions, mutations and imports, audit generation, scientific calculations, JSON digests and browser caching, the desktop UI, the Ipso server and offline application, deployment and backup scripts, localization, documentation, and tests. Particular attention went to the recently changed species behavior and the distinction between tree identity, a sample, and a coppice shoot.

Verification performed:

- `python3 -m pytest -q`: **1,499 passed**, in approximately 249 seconds.
- `make test-js`: all JavaScript suites passed.
- Twelve additional isolated Python probes reproduced the behaviors described below. They used the test database and temporary digest/media directories, not application data.
- Two additional Node probes reproduced first-load cache invalidation and overlapping Ipso saves. The latter exercised the actual save handler using the existing app-shell test harness.

The extra probes asserted the current problematic behavior; their passing results confirm reproduction, not correctness. They were kept outside the repository during this review. Proposed regression cases are listed at the end of this report.

The local environment used Python 3.13, Django 6.0.4, django-allauth 65.15.1, and Gunicorn 23.0.0. The Docker image uses Python 3.14. I did not inspect deployed image contents, production data, the live reverse proxy, tenant administration, or host backup storage. There was no real-browser or real-device test and no power-failure experiment. Findings requiring those conditions are explicitly qualified. No application code or live state was changed during the original review.

Follow-up verification: the full Python suite passed **1,792 tests** on Python
3.14.10 after S4 (previously 1,787 after S3). All JavaScript suites passed with
the refreshed browser libraries. System/migration checks, hashed installation,
repeatable locks, package audits and the production-image gate passed. Actual
Gunicorn HTTP checks exercised fresh migrations, login/CSRF, HTTPS forwarding
and a tenant-specific OAuth redirect without contacting Microsoft. The image
has no package installers or test tooling. The image gate covers fixable
high/critical OS/library advisories; it is not an all-severity certification.
The fixes began with failing regressions. S1/S3 tests exercise the actual OAuth login/callback views with
Microsoft's HTTP responses mocked. The owner subsequently checked production
volumes and found no negatives, so C9 requires no historical-data repair or
migration. The completion notes do not establish deployed package versions or
retest the remaining findings against production.

### Priorities

**P1** means address soon because ordinary workflows can produce incorrect durable data, access-policy surprises, or incomplete recovery. **P2** means a bounded defect or a risk requiring malformed input, interruption, or a particular deployment condition. **P3** means maintainability or an improvement to undertake when useful. These are application priorities, not CVSS scores.

Fixed rows retain their original priority for historical context. Open rows
carry the current priority. **Deferred** means intentionally postponed, not
fixed. SQLite-only work, including R3 and SQLite-specific parts of R2/I7/I8,
is P3 pending the probable PostgreSQL/PostGIS move. C12 is separately deferred
because of the current single synchronous worker; its file-publication race is
independent of the database backend. R1 and the general recovery requirements
in R2 keep their priorities.

| Priority | Finding | Status | Evidence / completion |
| --- | --- | --- | --- |
| P1 | C1 Multipart observation retries bypass deduplication | Fixed | `5fe9f3b` |
| P1 | C2 Successful writes and retry receipts are not atomic | Fixed | `d0db238` |
| P1 | C3 Rejected sampled-tree requests create parent samples | Fixed | `8dcb225` |
| P1 | C4 Intervento target edits can strand existing marks | Fixed | `086729c` |
| P1 | C5 Ceppaie per hectare counts shoots as stumps | Fixed | `4eab0cd`, `1356c3c` |
| P1 | C6 Another reader can retain a same-second stale digest | Fixed | `c570e21` |
| P1 | C8 Overlapping Ipso saves can capture duplicates | Fixed | `0eb6d2c`; Ipso-only |
| P1 | S1 Existing OAuth links outlive changes to login policy | Fixed | `3c2026a` |
| P1 | R1 Live backups can omit photographs referenced by their database | Fixed | `6f92a55`; coordinated capture and photo-reference/checksum validation |
| P2 | C7 Invalidations do not fence a first in-flight fetch | Fixed | `396900f` |
| P2 | C9 Positive measurements can yield persisted negative volumes | Fixed | `f202c92`; no production negatives reported, no repair migration needed |
| P2 | C10 Input parsing coerces invalid types and can raise exceptions | Fixed | `3130ca3` |
| P2 | C11 Some writes commit before invalidating their digests | Fixed | `6a7ceb8` |
| P3 | C12 Concurrent digest generators can publish an old result as fresh | Deferred | Revisit before adding workers/threads or concurrent generators |
| P2 | C13 Audit events within one minute are not reliably newest first | Fixed | `ce639ab`; full timestamp ordering, stable event IDs and precise browser/CSV data |
| P2 | C14 Account administration lacks a stale-form guard | Fixed | Versioned edits, email validation and last-active-admin protection |
| P2 | S2 Desktop observation photographs lack application byte limits | Fixed | `47baa8b`; shared stream/decoder limits and 1024 px normalization |
| P2 | S3 Database OAuth tenant configuration is not guarded at authentication | Fixed | `4b11f46`; environment-only app discovery and runtime tenant validation |
| P2 | S4 Dependency versions need a reproducible security-update policy | Fixed in branch | Upgraded locks/manifests, pinned base, inventories and automated checks; GitHub activation remains |
| P2 | R2 Restore documentation and replacement sequencing need hardening | Open, partly addressed | Photo restore docs/validation and unique archive publication done; broader maintenance/mirroring remains |
| P3 | R3 SQLite acknowledged writes are not durable across every host failure | Deferred | SQLite-specific; revisit durability against the PostgreSQL target |
| P2 | R4 Ipso can record more rows than its upload API accepts | Open | Source review |

## Correctness and data integrity

### C1 Multipart observation retries bypass deduplication

**Status: Fixed — `5fe9f3b`.** Multipart requests now replay completed nonces,
including observation/photo submissions. C2 supplies the shared request-identity
and atomic receipt contract.

Sources: [nonce middleware](../apps/base/middleware.py), lines 75–103; [observation save](../apps/bosco/views.py), `observation_save_view`; [form transport](../apps/base/static/base/js/forms.js), `postFormData`.

The form transport attaches a nonce to multipart submissions, and the success response records it. However, the replay middleware only examines POSTs whose content type is exactly `application/json`. Observation saves use multipart forms, so they never consult that recorded receipt.

**Reproduction:** submit the same valid multipart observation twice with the same user and nonce. Both requests return 200 and create different Observation rows. Photographs are not required to trigger the problem. Retrying an existing-row edit can instead encounter a version conflict rather than replay the successful response.

This defeats the retry guarantee precisely on a route where photographs and unreliable connections make retries useful. Support the same idempotency contract for JSON and multipart requests. For multipart requests, include normalized form fields and file checksums in the request identity without buffering an unlimited body.

### C2 Successful writes and retry receipts are not atomic

**Status: Fixed — `d0db238`.** A lazy write transaction claims the nonce and
commits the mutation, dirty marks, and replay receipt together. Receipts are
bound to user, operation, and payload; failures roll back and remain retryable.

Sources: [response helpers](../apps/base/responses.py), `save_model_response` and `success_response`; [nonce persistence](../apps/base/middleware.py), `save_nonce`; [mark save](../apps/piano_di_taglio/views.py), `mark_save_view`.

Several views commit their domain transaction before calling `success_response`, which persists the nonce in a separate transaction. A process failure, database failure, or exception in that interval leaves a committed operation without its replay receipt. Checking a nonce in middleware also does not claim it against a second simultaneous request.

**Reproduction:** inject a failure into nonce persistence after a new mark has committed. The first request returns 500 but one TreeMark exists. Retrying the identical request after removing the injected failure returns 200 and creates a second TreeMark. The request omitted a number, exercising the normal automatic-number path.

Make the mutation, its digest dirty marks, and its completed retry receipt one database unit. Claim the request within that unit using a uniqueness constraint; a competing request must not execute the mutation independently. Bind receipts to the user, operation, and payload identity. The current nonce lookup is not bound to an endpoint or request body, so accidentally reusing a nonce can replay an unrelated response rather than identify a mismatch.

Do not implement this by wrapping photo decoding, network operations, or a whole long request in a SQLite write transaction. Parse and validate expensive inputs first, then perform a short transactional operation. Ipso's session/content identity provides a useful existing model to examine, although its workflow differs.

### C3 Rejected sampled-tree requests create parent samples

**Status: Fixed — `8dcb225`.** Read-only validation precedes parent creation;
sample/tree creation and conflict handling share the write transaction.

Source: [sampled-tree save](../apps/campionamenti/views.py), lines 202–325 and `_find_or_create_sample` at line 1258.

`tree_save_view` calls `_find_or_create_sample` before validating the selected species and several identity/duplicate conditions, and before entering its write transaction. That helper can create a Sample. A later validation response does not undo this write or invalidate the relevant digests.

**Reproduction:** post otherwise valid unstructured sample data with a nonexistent species ID. The response is 400, but a Sample exists for the survey; no Tree was created.

An unsuccessful measurement can therefore leave a sample/date behind and affect subsequent grouping or summaries. Validate the read-only context first; create or resolve the durable parent inside the same transaction as its children. Check conflicts before creating anything. Tests should assert that rejected operations leave both database counts and digest state unchanged. Returning an error response from inside `transaction.atomic()` is not, by itself, a rollback instruction.

### C4 Intervento target edits can strand existing marks

**Status: Fixed — `086729c`.** Target edits validate existing marks against the
shared parcel/region scope rule and reject incompatible changes.

Sources: [item save](../apps/piano_di_taglio/views.py), around line 475; [plan item validation](../apps/piano_di_taglio/plan_item_validation.py), line 30; [mark scope validation](../apps/piano_di_taglio/mark_import.py), `mark_parcel_matches_item`.

The item-edit invariant protects linked harvests, but not existing TreeMarks. New marks are checked against the item's parcel/region; changing the parent target can invalidate marks that were valid when entered.

**Reproduction:** create a parcel-targeted item with a mark in parcel A, then edit the item to target parcel B. The save succeeds, the mark remains in A, and the shared mark-scope predicate now rejects that mark for its own item.

Before saving the proposed target, verify that every existing mark remains within it. Widening a parcel target to its containing region may be valid; narrowing a region target may be valid only if all marked parcels fit. Reuse the same scope definition across manual edits and imports. Reject an incompatible change with a useful explanation; do not silently move the marks or their tree identities.

### C5 Ceppaie per hectare counts shoots as stumps

**Status: Fixed — `4eab0cd`, `1356c3c`.** The calculation counts distinct
`(sample_id, tree_id)` stump observations, with each structured sample visit
where coppice stumps were observed contributing its area once. A later selected
survey contributes its own stump count and area. Standards-only sampled plots
are excluded from the stump-density denominator.
The fix includes refreshing the affected stored digest and regression coverage.

Sources: [stump density](../apps/base/digests.py), `_coppice_stumps_per_ha` at line 408; [sample creation](../apps/campionamenti/views.py), `tree_save_view`; [database model documentation](database.md), Tree and TreeSample.

The model deliberately represents a coppice stump with one Tree and its measured shoots with multiple TreeSample rows. `_coppice_stumps_per_ha` increments its count once per TreeSample, not once per stump in a sample. The denominator counts the sample area once, so the numerator is inflated by the number of measured shoots.

**Reproduction:** one coppice Tree, three shoots, and one structured sample plot produce `3 / plot_area_ha`, not `1 / plot_area_ha`.

Count distinct `(sample_id, tree_id)` pairs, retaining the documented selection of surveys and sample areas. Counting each Tree only once globally would also be wrong when the denominator includes repeated visits. Add fixtures with several shoots on one stump, several stumps, repeated surveys, and plots containing only high-forest standards. The existing docstring's reference to counting coppice rows should be corrected alongside the implementation.

### C6 Another reader can retain a same-second stale digest

**Status: Fixed — `c570e21`.** Digest responses identify published gzip content
with ETags, and the browser cache revalidates with `If-None-Match`. Date-only
validators cannot return a stale 304 for a digest.

Sources: [digest serving](../apps/base/digests.py), lines 154–206; [conditional HTTP helper](../apps/base/http.py), `not_modified_response`.

Digest validation uses Last-Modified dates with whole-second precision. Forcing a 200 response when the current request regenerated a digest fixes only that request. A different client can still present the old timestamp after another reader has regenerated the changed file within the same second.

**Reproduction:** client A receives an old digest and its Last-Modified value. A mutation dirties it. Client B regenerates changed content with the same whole-second timestamp. Client A revalidates with its old value and receives 304.

The stale content can survive every subsequent revalidation until a later generation changes the timestamp; this is not just one delayed refresh. Use an ETag based on the published content or a durable published-generation identifier, and have the cache send If-None-Match. A dirty counter alone is insufficient unless the response identifies the generation actually published.

### C7 Invalidations do not fence a first in-flight fetch

**Status: Fixed — `396900f`.** Cache revisions outlive individual entries and
cover registered/in-flight IDs, including cold loads and evictions. Obsolete
responses are discarded and initial loads retry against the current generation.

Source: [browser cache](../apps/base/static/base/js/cache.js), `takeSnapshot`, `snapshotIsCurrent`, and `applyResponseChanges` at lines 103–176.

Invalidations iterate existing cache entries. A first load has no entry yet, so invalidating that data ID or its prefix does not change the fetch snapshot. An old response can then populate the cache after the write that invalidated it.

**Reproduction:** begin an uncached load, apply an explicit invalidation, then resolve the old request. Its response is accepted. The existing revision protection does work for populated entries and optimistic patches; this is an uncovered absent-entry case.

Maintain request epochs independently of cached data. Explicit and prefix invalidations must fence registered/in-flight IDs even when their entries are absent or evicted. Include overlapping first loads and eviction/reload sequences in the tests.

### C8 Overlapping Ipso saves can capture duplicates

**Status: Fixed — `0eb6d2c`; Ipso-only.** Saves acquire a synchronous in-progress
guard; IndexedDB writes enforce scoped number uniqueness and an open session.
Upload/export freezing checks the actual record snapshot. A durable insert is
acknowledged separately from fallible refresh/preferences/backup follow-ups.

Sources: [mobile save handler](../apps/ipso/static/ipso/pwa/app.js), lines 1885–1968; [IndexedDB insert](../apps/ipso/static/ipso/pwa/store.js), `addTree` at line 381.

`onSave` awaits `numberExistsInSession` before disabling the button or setting its cooldown. Two invocations can both pass that asynchronous check. The store serializes its transactions and allocates different sequence numbers, but it does not enforce the domain number-uniqueness rule inside the transaction.

**Reproduction:** start two actual save-handler invocations and resolve both duplicate-number checks before either insert. Both handlers persist the same numbered record. This does not require broken IndexedDB sequencing.

Numbered duplicates can later make an upload fail. Blank-number trees are intentionally allowed, so an accidental duplicate of one can instead inflate a valid session without a duplicate-number warning.

Acquire an in-progress guard synchronously before the first await and release it in `finally`. Retain the cooldown as a UI convenience, not the correctness mechanism. Enforce scoped number uniqueness in the write transaction, and verify that the session is still open; another tab should not append records after the upload payload has been frozen.

There is a second acknowledgement hazard in the same handler: `Store.addTree`, session refresh, `localStorage.setItem`, form reset, and automatic backup share one catch block. A failure after the durable insert can say that saving failed and leave the entered values available for retry. Distinguish a failed record write from a successful write followed by a preferences/backup failure. This second case is source-established rather than separately reproduced.

### C9 Positive measurements can yield persisted negative volumes

**Status: Fixed — `f202c92`.** The agreed productive/harvestable-volume policy
clamps negative equation results to zero in Python and JavaScript. Positive
results retain normal rounding; unavailable estimates remain unavailable.
Source measurements are preserved.

Sources: [Tabacchi calculation](../apps/base/tabacchi.py), `_volume_from_b` at line 86; [mark volume and mass](../apps/piano_di_taglio/mark_import.py), line 190; [JavaScript calculation](../apps/base/static/base/js/volume.js).

The equations are evaluated for any measurements accepted by the request parsers, with no applicability-range or positive-result policy. Positive diameter and height therefore do not guarantee a physically meaningful result.

**Reproduction:** a mark for Ontano with diameter 5 cm and height 3 m is accepted and stores volume `-0.0055 m³`. Mass derived from that volume can also be negative.

The owner agreed that very small trees can have effectively zero productive
volume. Regression tests cover negative/zero and small positive estimates,
Python/JavaScript parity, and persisted volume/mass. The owner checked production
volumes and found no negatives; **no historical repair migration is needed**.

### C10 Input parsing coerces invalid types and can raise exceptions

**Status: Fixed — `3130ca3`.** Shared JSON decoding, typed field guards,
model-derived decimal bounds, nested-object checks, truth parsing, and choice
validation reject malformed inputs. Explicit derived/storage overflow also
rolls back writes. The existing localized 400 envelope is preserved, with UI
tests confirming entered values survive and forms can be corrected/retried.
CSV's intentional integral-number tolerance remains separate.

Sources: [numeric parsing](../apps/base/numparse.py), lines 94–145; [sample parsers](../apps/campionamenti/views.py), `_parse_tree_body` and `_parse_shoots`; [user save](../apps/impostazioni/views.py), line 627.

The shared integer parser promises to reject non-integers, but `int(value)` accepts JSON floats and booleans. Direct parser probes return 1 for both `1.9` and `True`; infinity raises an uncaught OverflowError. A large finite decimal coordinate such as `1e29` reaches quantization and raises InvalidOperation rather than producing an ordinary validation result.

Other parsers assume text values have `.strip()` or shoot entries have `.get()`. A malformed element can therefore become a 500. `_parse_shoots` uses Python `bool(...)` for one flag, making a string such as `"false"` true, unlike the shared truth parser. User role and login-method strings are assigned without explicit choice validation.

Use small shared, typed boundary validators: distinguish absent from invalid, reject booleans/fractional integer IDs, require dictionaries for nested records, validate choices, and check physical/storage ranges before decimal quantization. Return localized 400 responses for invalid input. Do not require a large schema framework to get these guarantees. Preserve intentional CSV tolerance separately from strict JSON parsing.

### C11 Some writes commit before invalidating their digests

**Status: Fixed — `6a7ceb8`.** Mutation and digest invalidation share a
transaction with or without a nonce. Imports/clears own their invalidation;
deletes recheck versions at the write boundary with a conditional write. The
completed fix retains SQLite-safe serialization; further SQLite-only locking
tuning is lower priority pending the backend move.

Sources: [sample deletion](../apps/campionamenti/views.py), lines 464–499; [hypsometric replacement](../apps/base/hypsometry.py), line 202; [hypsometric endpoints](../apps/impostazioni/views.py), lines 816–881.

Sample deletion calls `ts.delete()` before `mark_stale`, with no encompassing transaction. Hypsometric replacement/clear commits internally before its caller dirties the derived data. If the process fails between those steps, the database has changed while existing digest files remain marked fresh.

These are interruption hazards, not failures observed in a live instance. Their impact is nevertheless durable: no automatic check of the underlying tables repairs a digest that remains falsely fresh.

Make each domain mutation and all affected dirty marks one short transactional service operation. Its replay receipt belongs there too, as in C2. Keep irreversible filesystem effects separate and recoverable. Audit all delete/import/clear paths against the same contract rather than fixing only the two examples.

Version checking also needs a real concurrency contract. Some deletion paths check a loaded version before the delete, and `select_for_update()` elsewhere does not provide row locks on SQLite. Prefer conditional writes/deletes or another explicit serialization strategy where the check and mutation must be indivisible. [Django documents that select_for_update has no effect on SQLite](https://docs.djangoproject.com/en/6.0/ref/databases/#queryset-select-for-update-not-supported).

### C12 Concurrent digest generators can publish an old result as fresh

**Status: Deferred — P3 under the current single synchronous worker.** This is
not marked fixed. Revisit before adding workers/threads or allowing overlapping
HTTP/management-command generators. A PostgreSQL move alone does not address
the shared digest-file publication race.

Source: [digest generation](../apps/base/digests.py), `_regenerate_if_stale` at line 154 and `_write_gzip_json` at line 61.

The dirty-sequence compare-and-swap protects clearing the stale flag, but happens after publishing the file. It does not serialize competing generators.

**Reproduced interleaving:** generator A starts against old data. A write dirties the digest again. Generator B publishes new data and clears staleness. A then publishes old data; A's clear fails, but B already left the flag false. The final file contains old data and is considered fresh.

The current Docker command runs one synchronous Gunicorn worker, which substantially limits concurrent HTTP generation. This is a conditional risk involving another generator process or future worker/thread changes, not evidence that normal requests currently race in production. Unique temporary files and atomic renames already protect readers from partially written JSON; retain them.

Add per-digest cross-process generation ownership or serialization, and validate the generation before publishing. Immutable generation files plus an atomic published pointer are another option. Coordinate management-command generation with the same protocol. Do not add workers for performance before addressing this and the process-local security/cache assumptions.

### C13 Audit events within one minute are not reliably newest first

**Status: Fixed.** Audit generation sorts complete event timestamps with a
deterministic model/history-ID tie break. Composite event IDs remain stable
when rows or model configuration order change. The browser sorts fixed-format
UTC strings at microsecond precision, displays site-local seconds through a
separate hidden label, and exports precise UTC times. Migration 0030 forces
existing audit files to regenerate without changing history. Regressions cover
same-minute/sub-millisecond events, cross-model/exact ties, stable IDs, DST,
browser display/search/sorting/CSV and upgrade invalidation.

Source: [audit digest](../apps/base/digests.py), `generate_audit` at line 670 and `_audit_rows` at line 697.

Audit timestamps are formatted to the minute before the combined rows are sorted by that formatted string. Within a minute, the stable sort preserves each model's original ascending history order and its placement in the model configuration list, not descending event time.

**Reproduction:** create a harvest plan at second 01 and change its name at second 59 of the same minute. The digest displays creation before the later update even though its overall order is intended to be newest first. Precise timestamps remain in the history table; the digest loses the information needed to sort them correctly.

Sort using the full stored event timestamp and a deterministic event identity before formatting. Consider displaying seconds or exposing them in details. Add cross-model and same-second cases; do not rely on a generated row number as a durable audit-event identity.

### C14 Account administration lacks a stale form guard

**Status: Fixed.** Migration 0031 adds versions to users/history; account saves
advance the version except for login bookkeeping. Settings checks the submitted
version under account locks and returns the current row/form on a conflict;
the UI refreshes both without silently preserving stale fields. Email syntax is
validated server-side. Demoting/deactivating the last active administrator is
rejected, including competing requests; the actor's permissions are rechecked
under the same locks. Personal password writes update only the password field
so they preserve intervening admin changes. Failed/conflicted edits leave user
versions/history, whitelist entries, OAuth links/tokens, digest state and nonce
receipts untouched. Regressions include real overlapping SQLite requests and
the actual JS submit/modal flow. Target-backend locking remains a PostgreSQL
migration validation task.

Sources: [User model](../apps/base/models.py), line 129; [user save](../apps/impostazioni/views.py), line 627.

Unlike the application's timestamped domain models, User has no row version, and user saves do not compare one. An administrator's old form can overwrite another administrator's intervening role, email, active-state, or login-method correction. A database lock during save would not detect that the form itself was loaded earlier.

Add an explicit optimistic-concurrency token for account administration or a comparably clear conditional-update contract. Validate choices and email syntax server-side as well as in the browser. Decide whether demoting/deactivating the last active administrator should be rejected or require a deliberate recovery acknowledgement; there is currently no such guard in this save path. These are administrative integrity concerns, not a demonstrated ability for a non-admin to change a role.

## Security

### S1 Existing OAuth links outlive changes to login policy

**Status: Fixed — `3c2026a`.** Every callback rechecks the incoming provider
email against the current active OAuth whitelist, including provider-UID and
verified-email lookup paths. Settings edits remove superseded links/tokens in
the same transaction as identity changes. Matching subsequent logins can
relink; denials return a localized message. Callback and rollback tests cover
the policy, and the UI/docs explain relinking and Django's existing-session
semantics. Legacy stale links are protected without a data migration.

Sources: [OAuth adapter](../apps/base/auth.py), line 35; [user administration](../apps/impostazioni/views.py), `users_save` and `_sync_email_address`. The installed allauth account lookup and login flow were also inspected.

`pre_social_login` immediately accepts the existing-account path without rechecking the current login method or the whitelist email. Changing a user to password login updates the User and EmailAddress rows but leaves its SocialAccount link intact. Allauth resolves an incoming provider UID to that existing link before calling the adapter.

**Reproduction:** create a linked Microsoft OAuth user, change the user to password login through the admin API, then resolve the same provider identity through allauth. The old SocialAccount remains, lookup resolves the now-password user, and the adapter returns without rejecting the login policy mismatch.

The result is access through a formerly linked identity after an administrator has changed the account's permitted login method. Email reassignment has the same policy question. This is not arbitrary unauthenticated login: the incoming party still needs the linked external identity. It also does not demonstrate bypass of an inactive-user check.

Revalidate the login policy for existing social links on every login. Define unlink/relink behavior when changing a login method or identity email, and make the admin UI explain the consequence. Test actual callback completion as well as adapter lookup. Keep account deactivation and session revocation semantics explicit.

### S2 Desktop observation photographs lack application byte limits

**Status: Fixed — `47baa8b`.** Both desktop and Ipso photo paths now enforce actual streamed
bytes/count, per-observation photo counts, and decoded-pixel bounds. Fully
decoded images become proportional <=1024 px JPEGs at quality 85; orientation and
relevant metadata are preserved. Ipso also prepares the smaller image before
transmission. Staged mobile checksums remain stable for retries/import identity.
Malformed/oversized inputs leave no domain rows or nonce receipt; desktop errors
preserve the pending form. Regression tests use genuine images and bounded
streams, including false size claims. Defaults and operator-facing behavior are
documented in [Osservazioni](page-osservazioni.md#image-policy-and-upload-limits).

Follow-up — `35a1be6`: PWA 1.5.4 validates the published photo limits before finalization freezes
the upload payload, leaving rejected sessions editable with local photos intact.
Staged orientation-aware dimensions retain their wire identity for retries;
permanent dimensions are decoder-owned. Regressions cover both boundaries.

Related photo-display improvement — `3fe8ea9`: the Bosco modal and edit form now fetch real,
protected 320 px JPEG thumbnails instead of scaled-down originals. Derivatives
are cached on demand for both new and legacy images; full images remain linked
for explicit opening and PDF generation. Authentication, lazy legacy generation,
cache reuse, unchanged original bytes, and thumbnail-only browser requests have
regression coverage. Unique temporary files also prevent concurrent publication
collisions. The regenerable cache is separate from primary backup data.

Source: [photo parsing](../apps/bosco/views.py), `_uploaded_observation_photos` at line 571. Compare [Ipso upload limits](../apps/ipso/views.py), `_upload_too_large` and `_validate_upload_payload`.

The desktop observation path joins every uploaded file's chunks into memory and retains the resulting byte strings, without a per-photo or combined photo-byte limit. Checking Django's normal form-body limit is not enough: it excludes uploaded file data. [Django upload-limit documentation](https://docs.djangoproject.com/en/6.0/ref/settings/#data-upload-max-memory-size).

**Reproduction:** set the form-memory limit to 1,024 bytes and submit a 5,003-byte JPEG-signature file. The request succeeds and stores a photo larger than the limit. The path checks image signatures, not full decodability, so this probe need not contain a valid complete JPEG.

An authenticated writer or an accidental large gallery submission can consume the sole application worker's memory and media storage. A reverse-proxy limit could reduce exposure, but its actual configuration was not verified. This is an availability/storage finding, not a demonstrated image-decoder code-execution vulnerability.

Apply shared per-photo, photo-count, and aggregate byte limits while streaming, before whole-file buffering; return 413 when appropriate. Configure a matching total-request limit at the edge. If requiring fully decodable images, use bounded decoding and explicit pixel limits rather than adding an unbounded decoder to this route.

### S3 Database OAuth tenant configuration is not guarded at authentication

**Status: Fixed.** Microsoft credentials and tenant now come exclusively from
environment-backed settings. The social adapter overrides app discovery and
selection, never querying database `SocialApp` records. Login, callbacks and the
button share one configuration helper. Missing credentials or a broad/missing
tenant reject authentication locally before token exchange; Django's security
check reports invalid configuration as `base.E001`. Existing whitelist/link
policy remains in place.

The owner chose to retain the empty allauth configuration tables: deleting them
would require maintaining a custom upstream schema because `SocialToken.app`
references `SocialApp`. No database migration is needed. Regressions first
reproduced database app selection, competing configuration, unsuitable direct
login/callback tenants, and missing system checks; they now cover those paths
and query-free configuration discovery. See [security](security.md#ms-365-oauth-configuration).

Sources: [environment configuration](../config/settings.py), around line 119; [login-button availability](../apps/base/views.py), `_microsoft_oauth_configured` at line 91; installed allauth Microsoft adapter, `_build_tenant_url`.

Environment configuration rejects broad tenant aliases such as `common`. The UI helper also hides the login button for an unsuitable database SocialApp. However, that helper is not an authorization gate on the allauth login/callback routes. The installed provider accepts a database app's lowercase `tenant` override and otherwise has a broad default.

This is a misconfiguration risk, not a claim that the deployed tenant is broad or that a whitelist bypass was exploited. Button visibility should not be the mechanism enforcing an identity-provider trust boundary.

Validate the effective app/tenant used by authentication itself, including database SocialApps. Add a system check and fail closed on an invalid tenant. Document one supported configuration source or define its precedence clearly. Continue to treat provider identity and tenant policy as security decisions, not merely login-page presentation.

### S4 Dependency versions need a reproducible security update policy

**Status: Fixed in branch; GitHub activation/deployment remain operator actions.**
Production and test packages now have separate exact, hashed locks generated by
pinned uv tooling on Python 3.14. Installation enforces hashes; production also
requires wheels. The refresh includes Django 6.0.9, allauth 65.21.0, Gunicorn
26.2.0 and the native image/geospatial libraries. The first package audit failed
on Pillow 12.3.0; the deliberate move to Pillow 13.0.0 resolves those findings.
Final production/test lock audits and npm audit pass.

Browser dependencies have an npm manifest/lock; the vendor updater consumes
those sources and a regression-tested check rejects stale/missing/obsolete
served files. The official Python 3.14.10 Debian image is pinned by digest. The
image records actual Python/apt inventories and drops pip/ensurepip after build
validation, removing their unused bundled libraries flagged by the first image
scan. The final runtime inventory/native checks, Gunicorn smoke checks and
fixable-high/critical image scan pass. The latest development/build pip still
vendors older flagged libraries; monitor its upstream refresh separately.

Root GitHub configuration supplies monthly grouped Dependabot PRs for Python,
npm, Docker and pinned Actions, with advisory-triggered security updates enabled
through repository settings. CI runs tests, lock/asset consistency, package audits
and production-image checks/scanning on changes and daily. There is no manual
weekly inspection requirement or automatic deployment. Cold scans allow a
15-minute timeout; service failures fail the job. Repository notification/settings
have not been changed. The owner must merge to the default branch and enable
the features described in [dependency policy and GitHub setup](dependencies.md).

Python locks/base images are fixed, while apt security packages resolve at
build time. The recorded image ID and inventory identify each actual build;
these checks do not establish deployed versions or bit-identical OS rebuilds.

Sources: [requirements](../requirements.txt); [Docker build](../Dockerfile). Upstream notices were checked on the review date.

The inspected local Django 6.0.4 is behind later security releases, including 6.0.9 released on 6 October 2026. Some later issues involve features this app does not use; this is not proof that each advisory is exploitable here. The deployed package versions were not inspected. [Django October security release](https://www.djangoproject.com/weblog/2026/oct/06/security-releases/), [August security release](https://www.djangoproject.com/weblog/2026/aug/04/security-releases/).

`gunicorn>=23.0,<24.0` prevents upgrading into the currently supported releases listed by upstream. Upstream marks 23.0.0 unsupported; that does not itself establish an exploitable flaw in this configuration. [Gunicorn security policy](https://github.com/benoitc/gunicorn/security).

Broad requirements and a mutable base image also mean identical source revisions can build materially different deployments, while the local test environment can lag behind them. Use a tested dependency lock/constraints artifact, an explicit security-update process, and a generated image inventory. Pin the deployed image digest, not just the source revision. Separate test tooling from production requirements where practical. Deliberately evaluate supported Gunicorn releases rather than simply removing the upper bound during a deployment.

### Security controls worth preserving

The inspected code has substantial defenses already: role checks on writes, CSRF protection for session-based endpoints, closed self-registration, secure production cookie/HTTPS settings, bounded public Ipso uploads, protected media serving, fixed/allowlisted external and asset paths, safe DOM text insertion in many UI paths, and CSV formula hardening. Backup extraction validates member paths and rejects links.

I did not establish a working SQL injection, arbitrary-file-read, stored-script injection, or public role bypass in the reviewed paths. That is a scope statement, not a security certification. Check the deployed proxy's forwarded-header trust, body limits, timeout behavior, and access to the container port separately; the repository alone cannot establish those properties.

## Recovery and field reliability

### R1 Live backups can omit photographs referenced by their database

**Status: Fixed — `6f92a55`.** Media-mutating requests hold a shared cross-process lock
through their transaction/nonce completion and filesystem hooks. Backup takes
an exclusive lock while copying the database, primary photos and staged inbox,
then releases it before verifying/compressing private copies. Every photo
referenced by the captured database must match the copied path, size and checksum
before publication; restore repeats validation before exposing its destination.
The thumbnail cache is regenerable and excluded. Tests reproduce a real deletion
after the database snapshot, permit writes during compression, reject missing or
corrupt photos, and cover lock lifetime through commit hooks. The coordination
protocol remains relevant to PostgreSQL; the database-copy implementation must
be adapted with that migration. See [Backups](backups.md).

Sources: [backup creation](../bin/backup), `create_backup` at line 133; [observation deletion and photo replacement](../apps/bosco/views.py), around lines 324 and 647; [deployment sequence](../bin/deploy); [backup documentation](backups.md).

The SQLite backup API produces a consistent database snapshot. The script then archives the live inbox and observation-media directory. Observation edits/deletes remove photo files on transaction commit.

If a database snapshot includes a photo reference and the photo is deleted before tar reaches its file, the backup either fails or lacks media needed by the restored database. Atomic publication of individual files does not protect against this deletion window. Deployment backups are taken before stopping the application, so this is relevant to the supported workflow. SQLite integrity checking cannot detect a missing referenced photograph.

Either briefly quiesce mutations while taking the database/media snapshot, or retain immutable blobs long enough to cover in-progress backups and verify all database references against an archive manifest. Add checksum/reference validation and a restore drill containing edited and deleted photographs. Document the accepted consistency guarantee if live backups remain deliberately approximate.

This was established from the ordering of the code, not by interrupting a deployed backup. The inbox has analogous snapshot-timing considerations, but missing primary observation photos are the clearest data-loss consequence.

### R2 Restore documentation and replacement sequencing need hardening

**Status: Open — P2 for complete database/inbox/media recovery, validation,
maintenance locking, and rollback.** SQLite-specific database-file replacement
hardening is lower priority (P3) pending PostgreSQL/PostGIS. Preserve the general
staging, consistency, and recovery requirements in the target-backend workflow.

R1 also updates the manual restore instructions to include photos, validates
photo references before publishing a restore destination, and uses unique archive
temporary files with atomic non-overwriting publication. Those parts are done.
Full instance-maintenance serialization, interrupted mirroring/replacement and
operational/off-host recovery drills remain open.

Sources: [restore instructions](backups.md), Restore Sketch at line 158; [instance mirroring](../bin/mirror-instance-state), `RESTORE_SCRIPT` at line 29; [backup output naming](../bin/backup), `create_backup`.

The backup script includes observation-media, and its restore implementation supports it. The manual Restore Sketch still describes restoring the database and inbox without restoring observation-media. Following it can produce a working database with missing photographs. Update the runbook to restore and validate all three components.

The mirroring script replaces the target database before staging/replacing all its inbox and media files. A copy failure or interruption can leave a mixed generation. The stopped target and pre-mirror backup help recovery; they are real protections. Nevertheless, the script should stage and validate every component before replacement, retain the previous set until success, and provide an explicit resume/rollback procedure.

Operational commands also need serialization. Mirror staging paths are fixed, and two backups with the same instance/reason/second share an archive and temporary archive name. Concurrent runs can interfere. Use an instance-level maintenance lock and unique output/staging names, and refuse accidental overwrite of an existing finalized archive.

Recovery acceptance should include opening the app, checking audit/history, replaying pending inbox work, and verifying photograph checksums, not only `PRAGMA integrity_check`. Verify off-host backup availability and failure alerts operationally; this review did not inspect their deployment.

### R3 SQLite acknowledged writes are not durable across every host failure

**Status: Deferred — P3, SQLite-specific.** With a probable PostgreSQL/PostGIS
move, benchmarking/tuning SQLite's `synchronous` setting is lower priority.
No durability-setting change has been made. Establish the required commit
durability and verify the target backend's configuration during migration
planning; reconsider SQLite tuning if it remains the production backend longer.

Source: [database connection initialization](../apps/base/apps.py), line 21.

The connection uses WAL with `synchronous=NORMAL`. SQLite documents that this retains consistency in WAL mode but can lose committed transactions after power loss or an operating-system crash. An application-process crash has different, stronger durability guarantees. [SQLite synchronous documentation](https://sqlite.org/pragma.html#pragma_synchronous).

This is an explicit durability tradeoff, not evidence of corruption or observed lost data. Decide whether a successfully acknowledged field import or edit must survive a host failure. If so, benchmark production `FULL` and use it unless its cost is unacceptable. Record the recovery-point guarantee in the database and backup docs; nightly backups do not replace commit durability.

### R4 Ipso can record more rows than its upload API accepts

Sources: [configured maximum](../config/settings.py), line 54; [published upload limits](../apps/ipso/views.py), line 295; [upload validation](../apps/ipso/views.py), line 1973; [mobile session completion](../apps/ipso/static/ipso/pwa/app.js).

The server defaults to a maximum of 500 records per session upload. Reference data publishes the byte limit, not the record limit, and the app does not stop or roll over recording at that count. A legitimate long field session can therefore be completed into a frozen payload that the server rejects.

The records remain locally available; this is not demonstrated immediate loss of all measurements. It does create a difficult recovery workflow for an operator who has already finished collecting data.

Publish the limit and enforce an understandable session-rollover workflow before it is exceeded. If larger logical sessions are required, design stable-ID batching and resumability deliberately. Do not silently discard or renumber records to make an upload fit.

CSV backups also are not complete backups of observation photo blobs. The app already requests persistent browser storage and exposes related warnings; retain that work. Consider a full portable recovery export containing session metadata and photos, and test storage eviction/quota behavior on actual target browsers.

## Naming localization and maintainability

### M1 Scientific identity is coupled to editable localized names

Sources: [Tabacchi keys](../apps/base/tabacchi.py), lines 27–67; [JavaScript mirror](../apps/base/static/base/js/volume.js); [species editing](../apps/impostazioni/views.py), `species_save`; [species helpers](../apps/base/species.py).

Equation lookup uses `Species.common_name`, although most ordinary species names are editable. Renaming a supported species can turn future calculations into an unknown-species result. The Python table now obtains Abete/Faggio and the two catch-all names from the active string catalog, but many other keys remain literal Italian names; the JavaScript mirror also retains literal names. This mixes three different concepts: a scientific equation identifier, a persisted reference-data name, and a translated display label.

The current Italian installation's mirrored coefficients pass the existing parity tests. The issue is its fragility under edits and localization, not a claim that every current species calculates incorrectly.

Use a stable internal equation key, with an explicit proxy key for the two catch-all categories, while keeping display labels localizable. A small migration plus compatibility resolution for existing reference rows would be preferable to changing every historical tree name. Preserve provenance for materialized estimates rather than automatically rewriting historical measurements after a label or coefficient change.

This recommendation is separate from the intentionally narrow legacy-Altro exclusion. That exclusion now uses localized constants, remains independent of `active`/`minor`, and permits preserving an existing legacy assignment while rejecting new ones. I do not recommend adding a persisted selectability flag solely for that historical exception. The recent `strings_it.js` Ipso cleanup is present; the old filename issue is not a current finding.

### M2 Version and modification semantics are not uniform

**Status: Open for version/modification semantics.** C6 now identifies published
digest changes by content ETag; the per-model edit-version and `modified_at`
contracts remain to be clarified.

Sources: [timestamped model](../apps/base/models.py), `TimestampedModel`; [marked-volume materialization](../apps/piano_di_taglio/mark_import.py), `rematerialize_volume_marked`; [actual-volume materialization](../apps/prelievi/views.py), line 332.

Version increments are manually managed. Derived-volume updates use `save(update_fields=[...])`, changing stored data without incrementing the item version or updating its auto-now modification timestamp. Other workflows use a full save and explicitly increment the version. User and SiteSettings follow different contracts again.

A derived-only update need not necessarily invalidate an unrelated edit form, so blindly making every save increment a version would be an overcorrection. Define the intended contract: which fields participate in edit concurrency, whether derived changes update `modified_at`, and how a digest/cache validator identifies every published state. Provide small helpers or service operations reflecting those distinctions. Test them; comments claiming a universal write rule should match the exceptions.

### M3 Consolidate policy duplication before cosmetic duplication

**Status: Partly addressed.** C2/C11 centralize write/receipt/invalidation
boundaries, C4 shares target-scope validation, and C10 consolidates typed parsing
and validation responses. Broader service/module cleanup remains optional work.

There are useful shared components already: response envelopes, row builders, CSV utilities, species selection, domain import helpers, and shared diameter-class logic. Reusing row builders between patches and full digests is particularly valuable and should be retained.

The duplication most worth removing is validation and transaction policy: multiple integer/boolean/text parsers, separate manual/import scope checks, mutation-specific dirty lists, and inconsistent parent/version handling. These can produce different durable results for the same input. Small typed parsers and domain service functions would help more than a generic CRUD framework.

Python and JavaScript scientific calculation mirrors serve legitimate server-authority and offline/live-preview needs. Keep the parity tests. If coefficient tables keep growing, generate both copies from one neutral, versioned data source; do not remove the browser calculation just to eliminate duplication.

Internal naming mixes legacy Italian mobile keys such as `specie`, `numero`, and `particella` with newer English ID-based keys. Normalize at explicit serialization adapters and use consistent names within each layer. Do not opportunistically rename IndexedDB or queued-upload fields without a versioned migration: those names are part of an offline compatibility contract.

### M4 Reduce module size along existing domain seams

The largest modules combine form generation, parsing, domain validation, queries, imports, exports, cache updates, and filesystem operations. This makes write-boundary differences such as C3 and C11 difficult to spot. It also makes UI state ownership harder to understand.

Split by responsibility where behavior is already stable: request parsers, domain operations, query/row construction, import/export adapters, and UI controllers for independent detail/map/form sections. Ipso already has separate importers and staging helpers; build on that direction rather than duplicating it. Keep visible entry points and avoid circular dependencies between services and digest presentation.

Use the defect fixes to establish good service boundaries, then move code in separate behavior-preserving changes. Large simultaneous rewrites would make review and field regression diagnosis harder.

### M5 Documentation needs explicit exceptions and recovery guarantees

**Status: Partly addressed.** Completed fixes document multipart/JSON retries,
transaction/cache guarantees, numeric validation and the agreed volume policy,
linked OAuth identity/session behavior, image limits/renditions, and coordinated
database/media snapshots. The Restore Sketch now includes observation-media and
photo-reference/checksum validation. Broader maintenance/replacement sequencing,
operational recovery drills and remaining concurrency contracts still need
work. SQLite-specific durability documentation follows the lower-priority R3
decision.

The documentation is unusually useful and generally reflects the implementation. The gaps concern cross-cutting guarantees more than missing feature descriptions:

- Update the Restore Sketch to include observation-media and reference/checksum validation, as in R2.
- Document multipart mutation/retry behavior alongside JSON writes, rather than implying all writes have one wire format and guarantee.
- State which models use optimistic concurrency and which exceptions need a different contract. User administration currently has none.
- State the commit-durability and database/media snapshot guarantees explicitly; distinguish process failure from host failure.
- Document linked OAuth identity changes, not just first-time whitelist matching and button configuration.
- Preserve the documented legacy species-selection behavior, and distinguish label edits, defaults, equation proxies, and already materialized estimates.
- Describe the audit's scope accurately. High-volume tree/sample rows are intentionally excluded from detailed history; do not imply Controllo is a complete per-measurement ledger.

Prefer one authoritative cross-cutting document for each guarantee, with links from domain pages. Repeated prose about the same transaction/cache rule is likely to drift.

### M6 Modal accessibility lacks basic focus management

Sources: [modal implementation](../apps/base/static/base/js/modals.js); [shell markup](../apps/base/templates/base/shell_it.html), modal container.

The shared modal adds a plain div and an Escape handler but does not assign dialog semantics, move focus into the dialog, contain keyboard focus, make background content inert, or restore focus to the opener. This is a source-level accessibility gap, not a completed screen-reader audit.

Implement these properties centrally, with an accessible label and a sensible initial focus target. Test keyboard opening, tab traversal, Escape, validation errors, and focus return. Native controls and the existing safe text insertion can remain; this does not require a new UI framework.

## Improvements independent of defects

### I1 Measure and bound digest generation work

Sources: [digest builders](../apps/base/digests.py), `build_grid_record` at line 835 and `_hpi_turno` at line 1171.

Grid generation performs several related queries per grid: area count, region collection, survey count, and latest-area timestamp. Coppice item generation fetches ParcelPlanDetail per applicable item. Preload aggregates and relationship maps once per generator, while retaining a lightweight single-row builder for write-response patches. Add query-count tests at two collection sizes so linear N-plus-one growth is visible.

Measure generation duration, query count, uncompressed/compressed bytes, and peak memory using realistic copied or synthetic data. The current scale may make some queries harmless; measurements should decide which improvements are worth the complexity.

### I2 Keep audit cost proportional to useful work

Source: [audit generation](../apps/base/digests.py), lines 670–744.

Every dirty audit digest rebuilds the tracked history, materializes rows, sorts them, and serializes the whole result. Ordinary QuerySet iteration also retains its result cache. This will grow with the age of the installation, not just the current number of parcels or trees.

Start with streaming database iteration and benchmarks. If needed, use a paginated audit endpoint or an incremental projection keyed by real history-event identity. Preserve source history and its semantics; reducing audit coverage or deleting old history is not a performance optimization. Keep the existing check that every history-tracked model is explicitly accounted for.

### I3 Make mobile record operations scale without loading every blob

**Status: Performance work remains open.** C8 establishes transactional number
uniqueness and session/snapshot guards; preserve them in any later optimization.

Source: [IndexedDB store](../apps/ipso/static/ipso/pwa/store.js), `addTree`, `addObservation`, and `listTrees`; [save-time lookup](../apps/ipso/static/ipso/pwa/app.js), `numberExistsInSession`.

Insertion loads all existing session records to calculate the next sequence/count, and save-time duplicate detection loads them again. Observation records can carry photo blobs. Repeated recording therefore performs increasing work and copies unnecessary data on a resource-constrained device.

Use persisted sequence/count metadata for current-schema sessions, with a one-time legacy repair fallback. Use appropriate indexed lookups for scoped number identity, and metadata-only listing where photos are not needed. Consider storing blobs separately and retrieving thumbnails/photos lazily. Keep uniqueness and session-state checks transactional, and never reuse a sequence after deletion. Validate any IndexedDB migration against old open and pending-upload sessions.

### I4 Load desktop page code on demand

Source: [application boot](../apps/base/static/base/js/app.js).

Every domain page module is imported eagerly. The cost is paid even when opening a simple list or login-adjacent workflow. Profile cold transfer and parse time on the connection/device class the docs target, then load larger page/map/export modules on demand if the savings are material.

Keep route loading explicit and guard asynchronous mount results with navigation generations. Maintain authentication/logout cache clearing, service-worker asset/version rules, and predictable offline behavior. A full SPA framework migration is not a prerequisite.

### I5 Cache geometry preprocessing with explicit invalidation

Coordinate this improvement with the expected PostgreSQL/PostGIS work. Evaluate
the target spatial-query/index design before adding a separate persistent
geometry cache or custom spatial-index layer.

Source: [parcel geometry](../apps/base/parcel_geometry.py), `ParcelGeometryIndex`.

The index loads GeoJSON and parcel mappings, then checks feature bounding boxes linearly for each point. For larger imports, reuse validated immutable geometry indexes keyed to the geometry generation and relevant parcel mapping, or add a spatial candidate index if profiling warrants it.

Retain the current exact polygon check and deterministic handling of overlapping boundaries. A stale geometry cache can misassign parcels, so define invalidation and failure behavior before adding persistence. Benchmark realistic border points and imports, not just points near parcel centers.

### I6 Expose estimate coverage and provenance

Sources: [dendrometric export](../apps/base/dendrometry_export.py), line 35; [chart aggregation](../apps/base/static/base/js/dendrometry.js), around line 75; [parcel aggregation](../apps/base/digests.py), `generate_parcel_dendrometry`.

Missing volumes contribute zero to several totals, while those same trees still contribute to counts and basal area. Summing known volumes is a defensible policy, but a numeric total alone does not explain that it is partial. This is an interpretation risk, not a demand that unknown heights become mandatory.

Carry measured/estimated/unavailable counts and mark partial totals in charts and exports. Where useful, attach the equation/proxy and parameter-set identity to estimates. Keep precision and rounding policies explicit, and retain unknown-versus-zero distinctions at data boundaries. This makes the two new catch-all proxies clearer without changing their agreed scientific basis.

### I7 Make operating assumptions observable

Add structured timing/error logs for imports, digest generation, stale-generation
retries, database contention, rejected/oversize uploads, and backup reference
checks. Record a deployment's source revision and dependency/image inventory.
Alert on backup failure, media storage pressure, and repeated failed
field-session uploads. Detailed SQLite busy/WAL tuning is lower priority pending
the backend move; general operational visibility remains useful.

Before changing the one-worker deployment, assess the process-local rate limiter
and any cached mutable state, and implement C12. Keep write transactions short;
queue or precompute long report jobs only if measurements show that the
synchronous worker blocks ordinary use. Plan the probable PostgreSQL/PostGIS
move around spatial requirements, data/history preservation, and equivalents for
the current SQLite-specific triggers and constraints. Revalidate locking,
durability, and backup/restore behavior on that backend.

### I8 Add integration tests at the component boundaries

**Status: Partly addressed.** C1–C11, C13–C14, S1–S4 and R1 add boundary regressions for
writes, cache invalidation, offline saves/sessions, calculations, OAuth callbacks,
streamed image validation, renditions, and database/media backup consistency.
CI now runs the suites and verifies actual vendor assets and the production
image. Real-browser/device coverage and target-backend migration tests remain open.

The current test suite is extensive and valuable. The reproduced failures show that isolated tests and test doubles do not always model the boundary that matters: another digest reader, the first absent cache entry, a pre-save await, or a post-commit acknowledgement failure.

Add a small real-browser suite for navigation, modals, offline startup/update,
and save/upload/recovery on the supported mobile browsers. Prioritize
backend-independent boundary tests and migration/target-backend integration
tests preserving constraints, trigger behavior, and historical baselines.
Retain existing SQLite regressions; additional SQLite-only multiprocess/WAL test
infrastructure is lower priority. Where a SQLite locking case still needs
verification, use a file-backed database rather than an in-memory substitute.
Model network interruption and browser storage failures explicitly rather than
treating a successful mocked fetch as the only path.

## Suggested implementation sequence

Use small changes with their own regression tests and documentation updates.
C1–C11, C13–C14, S1–S4 and R1 are implemented; the remaining sequence is:

1. **Validate recovery operationally:** complete the off-host/app restore drills
   for the now-coordinated database/media backup; continue the open R2 work.
2. **Activate dependency automation:** merge the S4 configuration to the default
   branch, enable GitHub security notifications and run the workflow once.
3. **Strengthen recovery and field workflows:** the backend-independent parts
   of R2, plus R4 discoverable/enforced recording limits and recoverable sessions.
4. **Clarify scientific identity and estimates:** M1 stable equation keys and
   I6 partial-volume coverage/provenance, when justified by the use cases.
5. **Refactor and optimize selectively:** build on the completed shared
   boundaries and measured query/mobile costs. Coordinate geometry work with
   PostgreSQL/PostGIS planning; keep structural moves separate from behavior.
6. **Revisit deferred operational work when its conditions apply:** C12 before
   adding concurrent generators/workers; SQLite-only R3 tuning and related
   restore/test infrastructure remain P3 pending the backend decision.

Regression coverage and outstanding cases:

- [x] C1/C2: repeated multipart nonces, failures before receipt completion,
  concurrent claims, and nonce reuse with a different operation/body.
- [x] C3: rejected/conflicted saves leave no parent/tree or digest-state changes.
- [x] C4: plan target widening/narrowing with marks and linked harvests.
- [x] C5: several shoots/stumps, repeated sample visits, and standards-only plots.
- [x] C6/C7: same-second digest readers, invalidated first loads, and evictions.
- [x] C8: overlapping saves, transactional uniqueness, closed/frozen sessions,
  and fallible follow-up work after a durable insert.
- [x] C9: negative/zero/small-positive estimates, Python/JS parity, persisted
  volume/mass, and preservation of unavailable estimates. Production data was
  checked by the owner; no repair migration is needed.
- [x] C10: fractional/boolean IDs, malformed nested values, nonfinite/oversized
  numbers, boolean strings, unsupported choices, rollback, and UI-safe 400s.
- [x] C11: mutation/invalidation atomicity and delete-time version checks.
- [x] S1: actual callback completion after identity/method changes, inactive
  users, UID/email lookup paths, relinking, rollback, and existing-session rules.
- [ ] C12: competing publication generations; deferred under the single worker.
- [x] S3: environment-only discovery, ignored database apps, direct login/callback
  rejection for invalid tenants, and configuration security checks.
- [x] S4: hashed locks, repeatable resolution, asset-drift regressions, package
  audits, production tooling removal/inventory checks, and an actual Gunicorn
  migration/login/CSRF/HTTPS/OAuth smoke check; image gate passes.
- [x] S2: streaming byte/count and decoded-pixel bounds, genuine decoder inputs,
  normalization, metadata preservation, and stable staged upload identity.
- [x] C13: precise cross-model ordering, deterministic ties, stable identities,
  DST, browser sorting/search/CSV and upgrade invalidation.
- [x] C14: stale account edits, email validation, last-active-admin protection,
  overlapping requests, rollback and safe modal refresh/400 handling.
- [x] R1: deletion after DB snapshot, stable photo capture, writes during
  compression, missing/corrupt reference rejection, restore validation, and
  snapshot lock lifetime through nonce commit hooks.
- [ ] R2: interrupted instance replacement, full maintenance serialization and
  operational/off-host application recovery drills.
- [ ] R4: recording at/beyond the published session limit; upgrades with old
  open/pending sessions and photo blobs.

## Overall assessment

The code does not need wholesale replacement. Its strongest design choices are clear enough to preserve while repairing the boundaries: a server-owned data model, offline sessions with durable identities, common row/response formats, deliberate decimal handling, shared domain calculations, and extensive tests.

The completed fixes now cover rejected-save atomicity, retry receipts, stump and
volume correctness, digest/cache identity, mobile save acknowledgement, and
linked OAuth login policy and environment-only provider configuration. Image uploads now have server-owned limits/renditions,
and backup/restore validates every photo referenced by the captured database.
Operational/off-host recovery drills and broader maintenance sequencing remain.
Dependency upkeep is documented and automated after GitHub activation.
Continue with recovery validation, audit ordering,
and account-edit concurrency in small tested changes. Keep PostgreSQL/PostGIS
planning in view and give SQLite-only tuning lower priority than these shared
correctness and recovery requirements.
