# Settings page ("Impostazioni")

Path: /impostazioni

Collapsible sections are collapsed by default. Sections not visible to the
current user's role are hidden entirely. The tab is visible to every
authenticated user because every user can configure a landing page.

## Personal settings

Password-auth users see `Cambio password`, with two fields: new password and
repeat new password. OAuth users do not see the password section.

Every user sees `Pagina iniziale`. `Pagina personale` accepts a blank value or
a same-site app URL, such as `/prelievi`, `/bosco`, or
`/campionamenti?grid=1`; blank means use the site default. Admins also see
`Pagina default`, used for users without a personal value. External URLs and
non-app paths such as `/api/...`, `/admin/...`, and `/ipso/...` are rejected.

Login and `/` redirect to the effective landing page: personal value, then site
default, then `/prelievi`.

## Reference tables

Tractors and species are visible to writers. Observation categories are
visible only to admins. The collapsible sub-sections have sortable tables.
Tables support add and edit but not delete; entities are deactivated via the
`active` flag. The `Solo attivi` checkbox is checked by default and filters the
table.

- Trattori: manufacturer, model, year, active flag.
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
  Tree-volume calculations use the same two reference species' Tabacchi
  equations as fixed proxies; changing a catch-all's density changes its mass
  calculation, not its volume equation. Legacy Altro has no Tabacchi equation.
  Recognized existing species are classified during migration. Custom taxa that
  cannot be recognized show `—` until a writer selects a type;
  this avoids silently guessing their botanical type.
- Categorie di osservazioni: name, sort order, and active flag. Active
  categories are sent to Ipso in `reference.json`. The default rows are `fitosanitario`, `viabilità`,
  and `rifiuti`.

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

## Future production (Produzione futura)

Visible to writers. Determines the data set used to compute per-parcel forecast
production ("prelievo previsto") in the Bosco page (see `page-bosco.md`).

It consists of a pulldown of available harvest plans. Exactly one can be
selected at a time. It defaults to the plan whose (start year, end year) overlap
the current year and whose end year is greatest. Robust in the event that no
harvest plan exists.

Changing this setting invalidates the `future_production.json` digest.

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

## Hypsometric parameters (Parametri ipsometrici)

Visible to writers and admins. A read-only table shows the active hypsometric
parameter set, with Importa / Esporta / Elimina controls and a `Calcola nuovi
parametri` panel that fits new coefficients from selected surveys. The selector
lists surveys only when they have measured non-coppice tree rows, and displays
that eligible-row count. Fully documented — behavior, the compute→accept flow,
the CSV format, and the served digest — in [`hypsometry.md`](hypsometry.md).

## Diameter classes (Classi diametriche)

Visible to writers and admins, immediately below Parametri ipsometrici. The
radio control selects one site-wide five-centimetre class convention used by
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

Migration 0031 initializes existing users and historical user rows at version 1.
Subsequent account saves advance the version, including personal password and
landing-page changes; login-time bookkeeping alone does not invalidate forms.
No account or audit event is deleted by the migration.

The edit form explains that changing the login method or email removes previous
Microsoft account links and stored tokens. Email comparison is case-insensitive,
so case-only edits preserve the links. A later Microsoft login can establish a
fresh link only with the current whitelisted email and an active OAuth user.
The callback checks this policy even for links created before these rules were
introduced. See `security.md` for the separate existing-session behavior.
