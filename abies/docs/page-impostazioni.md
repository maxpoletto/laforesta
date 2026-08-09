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
  `Minore` flag, and active flag. Minor species are grouped under a single
  `Altro` entry in Prelievi (see `page-prelievi.md`); the flag is editable
  here.
- Categorie di osservazioni: name, sort order, and active flag. Active
  categories are sent to Ipso in `reference.json`. The default rows are `fitosanitario`, `viabilità`,
  and `rifiuti`.

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
- Email (required; for OAuth, must match Entra account).
- Username (password only; auto-populated from email for OAuth). Hidden when
  OAuth is selected.
- Password (repeated; hidden when OAuth is selected).
- First name, last name.
- Role pulldown (Membro / Redattore / Amministratore).
- Active checkbox; only active users can log in.

The initial admin account is configured at server installation time. OAuth users
must be pre-added with a matching email to whitelist them.
