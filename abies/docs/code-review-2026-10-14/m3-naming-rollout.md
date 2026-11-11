# M3 naming transition: implementation and rollout ledger

Specification: [M3-NAMING-1](proposal-m3-naming-transition.md).
Ledger created: 1 November 2026.
State: **proposal only; implementation and deployment have not started**.

This file preserves the evidence needed across intervening commits/releases.
Fill entries from observed results, not intended behavior. Use `unknown` or
`not checked` when appropriate. No empty gate is implicitly satisfied.
Do not put credentials, operator names, observations, photos, raw payloads or
personal device identifiers here. Use short installation aliases.

## Source baseline and release roles

| Item | Recorded value |
| --- | --- |
| Reviewed application source | `472d2e36f6c9708be0364d88f82a01328446665d` |
| Source Ipso APP_VERSION | `1.5.6`; deployed value not verified |
| Source IndexedDB / reference / upload versions | `9` / `1` / `1` |
| Source desktop table schema | unversioned, treated as legacy schema 1 |
| L: last deployed legacy release | unknown; inventory before implementation rollout |
| T: first complete transition release | not assigned |
| B: retained executable recovery bridge | not assigned |
| C: cleanup release | not assigned; no cleanup authorized |
| Target naming epoch / storage / reference / upload / table | `2` / `10` / `2` / `2` / `2`, reserved by spec |

For each deployed release, record Git SHA/tag, server image digest, PWA version,
dependency-lock revision, supported contracts and source of retained assets.
Do not equate PWA APP_VERSION, IDB version and upload schema version.

## Implementation record

Add one row per reviewable commit. Include the spec section/P-ID and A-IDs.
An unrelated application release does not close an open migration work package.

| Date | Commit/tag | Spec revision / work package | Contracts/adapters affected | Verification / limitations |
| --- | --- | --- | --- | --- |
| 2026-11-01 | application baseline `472d2e3` | M3-NAMING-1 drafted | Proposed contracts only; no runtime changes | Documentation review/link checks; no rollout evidence |

### Work-package status

| Package | Status | Completing commits and evidence |
| --- | --- | --- |
| P01 Inventories/manifests/fixtures | Not started | — |
| P02 Accepted source and staged working data | Not started | — |
| P03 Version-2 server contracts and bridge | Not started | — |
| P04 Desktop controllers/tables/exports | Not started | — |
| P05 Ipso storage/reference/UI/preferences | Not started | — |
| P06 State conversion and diagnostics | Not started | — |
| P07 Remaining symbols/package/assets | Not started | — |
| P08 Rehearsal and T deployment | Not started | — |
| P09 Intermediate releases and observations | Not started | — |
| P10 Retained B and cleanup C | Not started | — |

## Deployment record

Record actual deployment/activation, including rollback or failed attempts.
Store private backup locations in the operator's operational record; a safe
backup identifier/checksum can be referenced here.

| UTC date/time | Instance | Role T/T+n/B/C | Git/tag + image digest | PWA version | Server migrations/conversion/verification | Backup/recovery evidence | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Not deployed | Not inventoried | — | — | — | — | — | No evidence yet |

## Installation inventory

Count each origin/browser profile/PWA with its own storage as an installation.
Include offline devices. The same phone on dev and prod has separate storage.
A missing log entry is not retirement. Explicit retirement requires the owner
to verify that all data/evidence has been accounted for before removing the
installation from the active set.

| Alias / origin | Browser/PWA platform | Last checked UTC | APP_VERSION | IDB / naming epoch | Preferences/cache verified | Unresolved IDs / legacy retry envelopes | Offline reload + record/export/upload/retry | Disposition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Inventory not performed | Unknown | Not checked | Unknown | Unknown | Not checked | Unknown | Not checked | Blocks cleanup |

The inventory must be explicitly declared complete, by an accountable operator,
before G01 can pass. A table containing only checked devices is insufficient if
another known installation has not been examined.

## Server and data verification snapshots

Attach or reference sanitized check-only/apply/verify reports. Record at least:

- Command/release/UTC time and instance; maintenance/write exclusion used.
- Live mode/key conversions needed/completed and remaining collision counts.
- Number of retained uploads by source schema and state, verified source
  checksums, canonical projections, missing/corrupt source or working documents.
- Versioned digest/dirty-status identities, generation jobs and canonical patch
  coverage, including dynamically generated families.
- Normal nonce TTL and last legacy mutation time; outstanding legacy receipts.
- Remaining adapter traffic by A-ID/protocol/family/outcome, with no raw data.

No reports recorded yet. All counts are **unknown**, not zero.

## Temporary adapter register

Replace `not implemented` with exact module/function locations when adding code.
Record later moves, removal commits and recovery-only destinations. An adapter
must not disappear merely because a caller was renamed.

| Adapter | Live code location | Introduced / moved | Retirement evidence | Removed from normal runtime | Recovery destination |
| --- | --- | --- | --- | --- | --- |
| A01 IDB upgrade | Not implemented | — | — | No | B, to be retained |
| A02 Preferences | Not implemented | — | — | No | B |
| A03 Reference/geometry/history | Not implemented | — | — | No | B |
| A04 Schema-1 upload/checksum | Existing legacy implementation; bridge not yet isolated | — | — | No | B |
| A05 Source-to-working projection | Not implemented | — | — | No | B |
| A06 API/table/form/response projections | Not implemented | — | — | No | B |
| A07 Name-only identity fallback | Existing behavior; bridge not yet isolated | — | — | No | B |
| A08 Field/mode/counter aliases | Not implemented | — | — | No | B |
| A09 Internal geometry compatibility | Existing `parcelNames`/grid shape; bridge not yet isolated | — | — | No | Source-file parsers retained separately |

## Observation window

| Item | Evidence |
| --- | --- |
| Complete active/retired installation inventory | Not checked |
| Last legacy desktop mutation UTC | Unknown |
| Current nonce TTL / resulting earliest eligible cleanup time | 24 hours in source baseline; deployed policy and timestamp not checked |
| Start/end of observation window | Not started |
| One normal offline/record/upload cycle for each installation | Not checked |
| Last legacy adapter use, explanation and reset of window if required | Unknown |
| Intermediate releases preserving the compatibility matrix | None recorded |

## Cleanup gate evidence

Copy the exact criteria from specification section 12 when checking gates;
do not replace a criterion with a weaker proxy such as “no recent errors”.

| Gate | Status | Dated evidence / accountable check |
| --- | --- | --- |
| G01 All installations migrated/retired with data accounted for | Not checked | — |
| G02 No legacy mutable keys or unresolved required identity fallback | Not checked | — |
| G03 No remaining schema-1 retry source in any recovery flow | Not checked | — |
| G04 Every retained staged upload has verified source and canonical working data | Not checked | — |
| G05 Clients/nonce TTL/normal field cycle/legacy traffic window | Not checked | — |
| G06 Canonical caches/patches/jobs; no v1 publisher can return | Not checked | — |
| G07 Exact executable B retained; device and old-backup recovery rehearsed | Not checked | — |
| G08 Adapter dispositions and C compatibility/rejection tests | Not checked | — |
| G09 Owner/operator acknowledgment and authorization of C | Not requested | — |

## Bridge retention and recovery rehearsal

| Item | Evidence |
| --- | --- |
| B Git tag/SHA and retained executable image digest | Not assigned |
| Locked dependencies, full PWA assets, runtime requirements retained | Not checked |
| Retained specification/fixtures/migration instructions | This proposal only; executable bridge not built |
| Pre-T device upgraded through B on the same origin/profile | Not tested |
| Pre-T database/media/inbox restored through B on isolated instance | Not tested |
| New canonical upload plus original-v1 lost-response retry after recovery | Not tested |
| Recovery from an interrupted state/projection conversion | Not tested |
| Procedure for serving a complete B shell after C without clearing storage | Not rehearsed |

## Cleanup record

No cleanup has occurred. Before C, record the release and all passing gates.
After C, record the removal commits per A-ID, final tests, deployment checks,
remaining permanent human-file adapters and archival/framework exceptions.
Keep this ledger and the specification after removal of compatibility code.

## Decisions and deviations

| Date | Decision / changed spec section | Reason and compatibility effect | Review acknowledgment |
| --- | --- | --- | --- |
| 2026-11-01 | Prefer mutable-storage conversion plus bounded adapters over indefinite dual naming | Agreed design; implementation awaits review of the full specification | User requested this reproducible proposal |
