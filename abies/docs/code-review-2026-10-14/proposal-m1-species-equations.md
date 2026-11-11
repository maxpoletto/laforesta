# M1: species names, CSV imports and equation identity

30 October 2026. Reviewed revision: `8d0aaa98e30ec1d00de93f66f49262e73282a37d`.
Discussion and proposal only; no application changes have been made.

## The two kinds of equation need different identities

Human-readable CSV names are compatible with stable internal equation keys.
The original review should have distinguished height regressions from volume
equations more clearly:

| Concern | Current identity | Proposed identity |
| --- | --- | --- |
| Species in a hypsometric CSV | `genere`, matched against `Species.common_name`, ignoring case | Keep readable names at this import boundary |
| Persisted height regression | Foreign keys to parameter set, region and species | Keep these foreign keys |
| Height lookup in desktop/mobile UI | Region/species display names | Prefer region/species IDs in internal lookups |
| Tabacchi volume equation | Editable species common name | An explicit equation key on the Species row |

[The CSV parser](../../apps/base/hypsometry.py), `parse_param_csv()`, resolves
`compresa` and `genere` once, into actual Region and Species objects.
[HypsoParam](../../apps/base/models.py) then stores `region_id` and `species_id`;
it does not store an independent species-name string that must remain manually
synchronized. Its uniqueness key is `(param_set, region, species)`.

For example, a CSV can continue to contain:

```csv
compresa,genere,funzione,a,b,r2,n
Capistrano,Abete,ln,7.0306,-4.2563,0.6312,48
```

The importer resolves `Abete` to Species ID 17, for example, and saves that ID.
Separately, Species 17 might select volume equation `silver_fir`. The CSV does
not need to contain `silver_fir` or the database ID. Renaming that species later
does not break the persisted regression relationship.

## What actually breaks today

1. [Python Tabacchi lookup](../../apps/base/tabacchi.py) and
   [its JS mirror](../../apps/base/static/base/js/volume.js) use the common name as
   their coefficient-table key. A pure-Python check confirms that `Faggio` is
   recognized and produces `0.7022 m³` for D=30 cm and h=20 m, while
   `Faggio europeo` is not recognized. Keeping the same Species ID and Latin
   name does not help that lookup.
2. [The desktop height form](../../apps/piano_di_taglio/static/piano_di_taglio/js/piano-di-taglio.js),
   `findHypsoParams()` / `wireHypsoAutoH()`, matches displayed region/species
   names. [The hypso digest](../../apps/base/digests.py), `hypso_param_row()`,
   contains those names but no region/species ID columns.
3. There is also a specific cache dependency gap: renaming a species through
   [`_save_species_response()`](../../apps/impostazioni/views.py) does not dirty
   or invalidate `hypso_params`. If that digest was already generated, it can
   retain the old label while a newly rendered species option has the new label.
   Source inspection shows that those two strings will no longer join. This
   cache scenario has not been reproduced against a running Django instance.
4. [Ipso's reference projection](../../apps/ipso/views.py), `_ipsometrica()`,
   publishes nested region/species-name keys, consumed by
   [`ipso.lookup()`](../../apps/ipso/static/ipso/pwa/ipso.js). A freshly generated
   reference bundle uses current names consistently, but names still serve as
   identity. Older sessions/references need compatibility treatment.
5. An old CSV containing a former species name will no longer resolve after a
   rename. A Tabacchi equation key does **not** solve this import problem. Today
   the operator must use the current name; that is an independent boundary rule.

## Recommended scope

### First: separate volume-equation selection from the display name

Add a nullable/blank `tabacchi_key` to Species, using a small explicit registry
of supported equations. For example, `silver_fir` and `beech` are identifiers,
not translated labels or editable scientific names. The existing coefficients
and their numeric behavior remain unchanged.

- Abete selects `silver_fir`; Faggio selects `beech`.
- Altra conifera also selects `silver_fir`; Altra latifoglia selects `beech`.
  Keep their own species IDs, densities, Pressler defaults and classifications.
- Altro and unsupported taxa have no equation unless deliberately configured.
- A newly created custom species starts with no inferred equation. If equation
  selection becomes editable, expose a labeled selector with an explicit
  “unavailable” choice and record changes in Controllo.
- Backfill known rows conservatively from the current recognized names and
  verified reference taxonomy. Do not infer equations from botanical type
  generally, or assume every Latin name maps to a supported equation. Report
  unresolved/custom rows for review rather than guessing.
- Include the key in the species data used by volume previews. Pass the key to
  Python/JS calculations; keep names for display. Preserve numeric parity tests.

This needs no new globally stable species-code scheme. Existing Species IDs
already identify records within an installation. The new key identifies a
coefficient vector, which multiple species can intentionally share.

### Second: retain readable CSVs and fix internal height joins

Add hidden `region_id` and `species_id` columns to the desktop hypso digest and
join by IDs. The compute preview and saved digest must still share one row
builder. Fix the species-rename invalidation dependency even with ID joins:
the displayed species label must refresh too.

For Ipso, introduce an ID-based reference lookup through a versioned/additive
adapter; retain the old `ipsometrica` representation while supported installed
clients still need it. Follow the offline compatibility sequence in
[the M3 audit](audit-m3-duplication-and-naming.md). No rename of IndexedDB keys
is necessary to introduce an internal ID lookup.

Continue importing/exporting current readable names. Resolve a name only when
it identifies exactly one row. In particular, case-insensitive matching must
not silently pick one of two names that differ only by case. Report unknown or
ambiguous names with the CSV row number and leave the current parameter set
untouched. Add aliases only if accepting files with old names becomes a real
requirement; an alias table is not required to fix volume lookup.

The volume proxy for a catch-all species must **not** imply a height proxy.
Altra conifera's height regression remains its own `(region, species)` row;
it must not silently borrow Abete's `a` and `b`.

## Tests before any implementation

- Rename a supported species: future volume remains available with the same
  coefficients, density and rounding; stored tree measurements remain unchanged.
- Import the same readable-name CSV before/after the identity change. Validate
  unknown and case-ambiguous names, and whole-import rollback.
- Rename after generating `hypso_params`: the current label and automatic
  height lookup both refresh correctly; the existing parameter FK is preserved.
- Two species can share a Tabacchi key without sharing height regressions.
- Unsupported species/Altro retain unavailable-volume behavior; zero and tiny
  positive estimates retain the C9 contract.
- Desktop and Ipso ID lookups agree; old reference bundles, resumable sessions
  and pending uploads remain usable through the compatibility adapter.

Do not recompute historical heights, volumes or masses as part of this change.
Adding per-tree equation/coefficient provenance would be a separate decision.
An equation key alone does not record which historical coefficient revision
produced an already-materialized estimate.
