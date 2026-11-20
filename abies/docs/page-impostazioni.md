# Settings page ("Impostazioni")

The default URL is `/impostazioni`. Writers and admins see three local
navigation links below the header on desktop and mobile. Each tab uses the
full content width.

| Tab | URL | Sections in display order |
| --- | --- | --- |
| Generali | `/impostazioni` | Password, Utenti, Pagina iniziale |
| Dati di base | `/impostazioni?scheda=dati-di-base` | Categorie di osservazioni, Tipi di viabilità and Classi economiche, then Specie, Parametri dendrometrici and Classi diametriche, then Mezzi |
| Cartografia | `/impostazioni?scheda=cartografia` | Region map, Ipso drawings and road publication |

Readers enter Generali directly. An unavailable or unknown tab returns to
the default URL. Switching tabs adds a history entry and preserves open
sections and form input while Impostazioni remains mounted. Sections start
collapsed on entry. Admins can open all three sections in the first group.

The plan used for projected production and the cutting calendar is selected
directly in [Bosco > Caratteristiche](page-bosco.md#mode-panels).

## Cartografia

The sidebar starts with **Compresa**, then shows **Particelle**, **Strade**,
and **Disegni**. Only **Disegni** can be collapsed. Each list grows with its
contents and scrolls after six rows. The eye icons control map visibility
for individual rows or the entire list. Drawings also have trash icons for
individual and bulk deletion. Bulk icons appear when their list has entries.

The **Compresa** dropdown at the top of the sidebar lists regions alphabetically.
It remembers the last selection when the operator returns to Cartografia in the
same browser tab. A link to a specific compresa opens that compresa.
**Nuova** and **Proprietà** open a form with a required name and optional
description. Saving a new name makes it available to Ipso on the next sync.
For a compresa containing only map data, **Proprietà → Elimina** lists how
many parcels, outlines, roads and drawings will be deleted. Plans, surveys,
observations, saved walks and pending field uploads protect the compresa.
If an unfinished upload received before the server upgrade has no region ID in
its saved records, deletion becomes available again once Abies can identify
the compresa on completion or the unfinished transfer expires.

Selecting a parcel or road enables its **Proprietà** button. Parcel properties
use the Bosco form. Renaming asks for confirmation that the existing records
follow the same parcel, and a parcel with dependent records is protected from
deletion. Road properties include the name, type and delete action.

Use **Confini** for parcel outlines or **Punti** for a road. Drag a vertex
to move it or click the small handle between vertices to insert one. Select an
existing vertex and use **Rimuovi vertice** on the map to remove it and reconnect
its neighbors. The broken-link icon opens an outline at the selected vertex.
Saving with an open outline asks for confirmation before leaving it out.
A parcel in use must keep at least one polygon. The undo button steps back
through the last ten actions in the current edit. Activate
**Poligono** during a parcel edit to sketch another piece. Click the first
vertex again to close the outline and add the piece to the edit. A coincident
edge shared by parcels in the compresa is edited on both sides and the affected
outlines are highlighted. Removing a vertex along a shared edge removes it
from both outlines. The editor keeps vertices needed for a valid shape or
where shared edges meet other edges. **Salva** publishes the edit and
**Annulla** discards it after confirmation. The unsaved edit survives switching
tabs within a browser session.

An imported Ipso walk appears as one drawing. **Modifica** lets the operator
select a trace or point, move a vertex, copy a side road into a new drawing,
split a line at a selected vertex with the broken-link icon, or remove parts.
The link icon lets the operator choose two line endpoints to join. Choosing
both ends of one line closes a polygon. **Promuovi** turns the resulting line
into a classified road immediately. Canceling the promotion returns to the
drawing with its edits intact.

**Nuovo** under **Disegni** starts a drawing in the selected compresa.
Choose **Linea** or **Poligono** on the map, then click or tap the map to
place vertices. Use **Termina** for a line, or return to the first vertex to
close a polygon. Abies then asks for the drawing's name. Canceling restores the
previous selection. During **Modifica**, the same tools add another shape to
the current drawing. The map tools are visible during drawing or editing. The
**Annulla** and **Salva** controls for an existing drawing are in **Disegni**.
Promoting a polygon opens Bosco’s parcel properties form. The first economic
class by name is selected, and the cadastral area can be left blank. In a
compresa with cataloged parcels, the destination selector also offers to add
the polygon to an existing parcel.

The trash icon above **Disegni** clears the working copies in the selected
compresa. **Importa… → Da Ipso → Già importati** reopens the edited copy if
one exists.
After deletion or promotion it restores the original field measurements, even
after inbox cleanup. The selected compresa uses the `c` query parameter and
the basemap uses `mt`.
The [cartography design](cartography.md) also describes additional import tools.

### Testing the road workflow

The browser test starts a local Django test server with a temporary SQLite
file so concurrent map requests use the same WAL mode as a running instance.
It records a short walk using simulated GPS fixes. At desktop width it opens
the upload from the Ipso inbox. At portrait width it imports the same kind of
walk from the editor. Both runs trim and join an interrupted walk, promote the road, check that
Bosco remembers the road filter after a reload and recover the original fixes.
With the Playwright Node module and its Chromium installed, the command is

```sh
ABIES_PLAYWRIGHT_MODULE=playwright python -m pytest -q test/test_cartography_browser.py
```

An absolute path to the installed Playwright module also works. The API
workflow and regression tests run as part of the regular `make test` suite.

## Personal settings

Password-auth users see `Password` with fields for the current password,
the new password and its confirmation. Admins also see `Utenti`.

Every user sees `Pagina iniziale`. `Pagina personale` accepts a blank value or
a same-site app URL, such as `/prelievi`, `/bosco`, or
`/campionamenti?grid=1`; blank means use the site default. Admins also see
`Pagina default`, used for users without a personal value. External URLs and
non-app paths such as `/api/...`, `/admin/...`, and `/ipso/...` are rejected.

Login and `/` redirect to the effective landing page: personal value, then site
default, then `/prelievi`.

## Reference tables

Tractors and species are visible to writers. Admins also manage observation
categories, road types and economic classes. Each collapsible section has a
sortable table with add and edit controls. Tables with an active flag also
have a `Solo attivi` filter, checked by default.

The trash icon appears beside unused economic classes, road types, and
observation categories. Selecting it opens a confirmation that names the
entry. Abies checks its version and assignments again before deleting it and
records the deletion in Controllo.

- `Mezzi` lists the vehicle name, manufacturer, model, year and active flag.
  The reference CSV importer accepts both the `Mezzo` column from the settings
  export and the existing `Trattore` column.
- Specie: common name, Latin name, density (q/m³; see `database.md`; used to
  derive per-tree mass for samples and marks), default Pressler coefficient,
  `Minore` flag, active flag, and one **Tipo** column with a
  **Conifera / Latifoglia** dropdown. Selecting a type saves immediately,
  using the row version to reject stale changes. Classification is also required
  in the add/edit modal. Changes are recorded in Controllo and invalidate the
  dependent digests/caches. Minor species are grouped by botanical type in
  Prelievi (see `page-prelievi.md`).
  `Altra conifera`, `Altra latifoglia`, and historical `Altro` all remain visible,
  including under `Solo attivi`. Their names and major-species status are
  protected. The catch-all types are shown as read-only text: Conifera,
  Latifoglia, and `—` for Altro, whose historical trees cannot be classified
  reliably.
  Density and default Pressler remain independently editable for all three.
  Initially, the new catch-all defaults copy Abete bianco (`Abete` / `Abies alba`
  in the canonical list) and Faggio, respectively: canonical density 9.00 / 10.50
  q/m³ and Pressler 2.00 / 2.00. These are initial values, not live links.
  Their volume equations initially reference the same two species through
  editable, explicit proxies. Changing a catch-all's density changes its mass
  calculation, not its volume equation. Legacy Altro initially has no volume
  equation; equation availability is visible and editable in the popup.
  Recognized existing species are classified during migration. Custom taxa that
  cannot be recognized show `—` until a writer selects a type;
  this avoids silently guessing their botanical type.
- Categorie di osservazioni: name, sort order, and active flag. Active
  categories are sent to Ipso in `reference.json`. The default rows are `fitosanitario`, `viabilità`,
  and `rifiuti`.
  An assigned category can be deactivated. Its edit form explains whether an
  observation or an incoming Ipso upload uses it. Accepted pages protect their
  categories as soon as they reach the server. For work still on operators'
  phones, deactivating a category keeps its ID available for later uploads.
- `Tipi di viabilità` lists the site-wide road and trail types. Admins can
  edit the name, optional description, sort order (`Ordine`) and active flag.
  The migration seeds these editable defaults.
  All road types are unused until the cartography editor can assign them to roads.

  | Nome | Descrizione | Ordine |
  | --- | --- | ---: |
  | Strada di 1° livello | Strada forestale camionabile | 10 |
  | Strada di 2° livello | Strada forestale adatta a macchine operatrici e automezzi 4x4 | 20 |
  | Pista | Pista da esbosco per trattori e macchine forestali | 30 |
  | Sentiero | Percorso per pedoni o animali | 40 |

- `Classi economiche` lists parcel classes by name. Admins can edit `Nome`
  (up to ten characters), an optional `Descrizione`, and the `Ceduo` checkbox.
  Renaming a class updates its label in Bosco and preserves parcel assignments.
  Once a parcel uses a class, its `Ceduo` checkbox is locked because coppice
  parcels require an intervention interval and a count of retained standards
  per hectare. Change an individual parcel's class and these values in Bosco.

  | Nome | Descrizione | Ceduo |
  | --- | --- | --- |
  | A | Fustaia di abete | No |
  | B | Fustaia mista di conifere e latifoglie | No |
  | C | Fustaia di pino | No |
  | D | Fustaia di pino nero e latifoglie | No |
  | E | Fustaia di faggio e conifere | No |
  | F | Ceduo di castagno | Sì |

### Species equation editor

The Species summary includes `Eq. altezza` (a check mark and the number of
comprese with an active regression) and `Eq. volume` (availability). Coverage in
one compresa does not imply coverage everywhere. Equation coefficients are
edited through the species pencil popup, alongside its existing parameters.

The popup has three groups:

1. Species parameters: common/Latin names, density, Pressler, botanical type,
   minor and active flags. Existing protected catch-all identity rules remain.
2. Volume: no equation, own coefficients, or another species' equation by a
   stable reference. Own coefficients are `b0`, `b1`, optional `b2`; the displayed
   formula documents D in cm, h in meters and volume in dm³ before conversion
   to m³. Source text is editable. Every configured equation is inspectable
   through its owning species, including explicit catch-all proxies.
3. Height: one enabled/disabled row per compresa, with `a`, `b`, optional N/r²,
   the logarithmic formula and retained origin information. Disabling a row
   removes only that species/compresa equation from the next active snapshot.

Saving is atomic: metadata, equation coefficients, history and invalidation
either succeed together or leave the form/data unchanged. A stale species
version or changed equation configuration returns the current form for review.
Invalid coefficients, missing proxy targets and proxy cycles are rejected.
An equation still needed by another species cannot be removed.

Changing one species or regional equation preserves all other species and
regions. Old height snapshots remain archived, and unchanged rows retain their
original provenance. Renaming a species updates labels without requiring a
second rename in an equation table. All runtime links use IDs. Equation edits do not recompute stored tree heights, volumes or masses.

Volume proxies share coefficients only: they never borrow height regressions,
density or Pressler defaults. A proxy follows later edits to its referenced
equation. Each height equation belongs to its own `(compresa ID, species ID)`
pair. The tree-entry form reports unavailable height/volume estimates; zero is
reserved for a calculated zero harvestable volume, including negative formula
results clamped to zero.

### Batch equation updates

The Species toolbar has two rows: `Importa equazioni:` and `Esporta equazioni:`,
each with `ipsometriche` and `di volume` buttons. Imports retain human-readable
species and compresa names in CSVs; database IDs are not required in the file.

1. Select a height or volume CSV. Abies reads it without changing settings.
2. Review a table of incoming names, matched species/compresa, current and
   proposed values, and row errors. Case-insensitive matching is automatic only
   when it identifies exactly one existing record.
3. Resolve unknown or ambiguous names with the provided selectors, then recheck
   the preview. For example, map an old `Abete` label to `Abete bianco`. Proxy
   references in volume files also require an unambiguous match. No species or
   compresa is silently created, and invalid or duplicate rows block confirmation.
4. Confirm to apply all reviewed rows atomically, or cancel to leave everything
   unchanged. Configuration changes during review, or a preview older than
   15 minutes, require a refreshed preview before confirmation.

Both imports update only the supplied equations. Omitted species and comprese
are preserved; a partial file never clears the rest of the configuration.
Removing a regional height equation is an explicit action in its species popup.
There is no global delete button. Height exports contain the entire active set;
volume exports include configured equations and explicit proxy references.
See [`hypsometry.md`](hypsometry.md#reviewed-batch-imports) for CSV formats.
Invalid coefficients name the field and its numeric or storage rule in the
preview. The preview remains read-only until all errors are resolved. See
[CSV imports](csv-imports.md) for shared field diagnostics and row numbering.

`Calcola equazioni di altezza`, inside the Species section, fits regressions from
selected surveys with measured non-coppice trees. Its preview shows `a`, `b`,
N and r² before acceptance. Acceptance recomputes and verifies the reviewed
candidate, updates only the resulting pairs, and preserves every other
equation. An optional checkbox uses those surveys for Bosco's height/diameter
chart. The panel describes the active snapshot and its source surveys; the
species popup retains each regional row's original provenance even when a
snapshot combines imported, computed and manual equations.

All accepted changes appear in Controllo, refresh dependent digests and evict
affected browser caches. Ipso receives the new height references on its next
reference refresh; already-recorded observations keep their measurements.
Repairing old unavailable/zero volume estimates is a separate one-time CLI
operation, outside the production settings UI.

### Species selectability and historical Altro

`active`, `minor`, and selectability have different purposes:

- `active` normally makes a species available for new entries.
- `minor` controls species grouping in Prelievi; it does not prevent selecting
  that species for a new marked or sampled tree.
- `Species.selectable` is a computed property, not a stored flag:
  `active and common_name != S.SPECIES_OTHER`. Historical Altro is therefore
  never a new-entry choice, even when active and not minor.

The shared `selectable_species()` query applies that same explicit exclusion
to species choices in Martellate, Alberi campionati and Bosco/PAI. Prelievi
additionally requires `minor=False`. Existing Altro records remain editable;
an existing Altro tree can also be measured again through the prior-tree
picker without creating a new tree identity. Save validation enforces these
exceptions server-side, so bypassing the pulldown cannot assign Altro to a
new tree or harvest. Historical CSV imports and saved Ipso uploads retain
their original species IDs/names rather than being relabeled.

Ipso reference data retains Altro metadata for historical/offline records,
with `selectable=false`. The mobile pulldown checks that flag and also excludes
`S.SPECIES_OTHER` by name, covering older cached references without the flag;
an old Altro sticky default is not restored as a new-tree selection.

All runtime Altro exclusion checks use the localized `S.SPECIES_OTHER` constant,
from `config/strings` on the server and Ipso's own string catalog on mobile.
This is deliberately a narrow legacy compatibility rule, not a general
database selectability flag. The translated constant must match the stored
`common_name`: switching string catalogs alone does not translate existing
species rows. Frozen data migrations retain their original names intentionally.
Fresh installations still include Altro as a compatibility reference row.
Of the three catch-alls, only Altra conifera and Altra latifoglia are offered
for new entries; historical exceptions matter only if legacy records exist.
Settings continues to show all three independently of selectability.

## Dendrometric data (Parametri dendrometrici)

Visible to writers. Determines the data set used to compute dendrometric
parameters in the Bosco page (see `page-bosco.md` > Dendrometria).

The top of the section is a line listing the current total number of trees,
regions, and parcels used for the dendrometric parameters.

Below is a multiselect of available structured surveys (Rilevamenti with a
sample grid). Unstructured surveys are not listed and cannot be active for
dendrometric purposes. This is intentionally stricter than the hypsometry
survey selector, which may list unstructured surveys with measured heights (see
`hypsometry.md`). The currently selected surveys are highlighted. Defaults to
the first structured survey by name alpha order, and is robust to there being no
structured surveys.

Below the multiselect is an `Aggiorna` submit button that causes the user's
selection to go into effect.

Updating this setting invalidates `parcels.json` (whose `Ceppaie / ha` value is
materialized from the selected surveys), `parcel_dendrometry.json`, and
`parcel_dendrometry_points.json`. The latter may also be driven by Parametri
ipsometrici when a computed parameter set opts the Altezze chart into its
source surveys. A successful save evicts those full client caches.

## Diameter classes (Classi diametriche)

Visible to writers and admins, below Parametri dendrometrici. The
radio control selects one site-wide five-centimeter class convention used by
every diameter-class chart and export:

- `centered`: class 20 contains integer diameters 18–22 cm;
- `shifted_up`: class 20 contains 20–24 cm;
- `shifted_down`: class 20 contains 16–20 cm.

`Salva` writes `SiteSettings.diameter_class_mode` and records the change in the
audit history. A changed value marks `parcel_dendrometry.json` and every
existing `mark_trees_<item-id>.json` / `sampled_trees_<survey-id>.json` stale.
The response also evicts those exact/prefixed in-memory caches. The raw
height/diameter points digest is unaffected because it contains no diameter
classes. Each affected digest is recomputed on demand on the next relevant
Bosco, Martellate, or Rilevamenti navigation; both ZIP CSV exporters read the
setting directly for the requested export.

## App users

This section is visible only to admins.

Admins can create and edit users. Table columns: first/last name, username,
email, login method, created-at, active.

Form fields:

- Login method radio (password / OAuth).
- Email (required, with syntax checked by the server; for OAuth, must match
  the Entra account).
- Username (password only; auto-populated from email for OAuth). Hidden when
  OAuth is selected.
- Password (repeated; hidden when OAuth is selected).
- First name, last name.
- Role pulldown (Membro / Redattore / Amministratore).
- Active checkbox; only active users can log in.

The initial admin account is configured at server installation time. OAuth users
must be pre-added with a matching email to whitelist them.

Account edits carry a hidden row version. Saving an old form returns a conflict
without changing the account, email whitelist, Microsoft links, or audit history.
The modal then shows the current account values and an explanatory message;
review those values before editing and saving again. Ordinary validation errors
keep the entered values available for correction. The users table also refreshes
to the current row on a conflict.

Settings rejects demoting or deactivating the last **active administrator**.
Another active user with the Abies administrator role must exist first; an
inactive administrator or Django staff/superuser flag does not count. Normal
name/email corrections to the last administrator are allowed. Concurrent account
edits are serialized so two administrators cannot each remove the other remaining
administrator. This policy applies to Settings; installation and authorized
server-side recovery tools can still provision an administrator.

Account saves advance the version, including personal password and landing-page
changes; login-time bookkeeping alone does not invalidate forms.

The edit form explains that changing the login method or email removes previous
Microsoft account links and stored tokens. Email comparison is case-insensitive,
so case-only edits preserve the links. A later Microsoft login can establish a
fresh link only with the current whitelisted email and an active OAuth user.
The callback checks this policy even for links created before these rules were
introduced. See `security.md` for the separate existing-session behavior.
