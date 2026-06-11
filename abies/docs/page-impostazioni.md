# Settings page ("Impostazioni")

Path: /impostazioni

Collapsible sections (all collapsed by default). Sections not visible
to the current user's role are hidden entirely. The tab itself is
hidden for users who would see no sections (reader + OAuth).

## Personal settings

Visible to password-auth users (all roles). Two fields: "new password"
and "repeat new password".

## Crews, tractors, and trees

Visible to writers. Three collapsible sub-sections, each with a
sortable table. Tables support add and edit but not delete — entities
are soft-deleted via the `active` flag. "Only active" checkbox
(checked by default) filters the table.

- Crews: name, notes (optional).
- Tractors: manufacturer, model, year.
- Trees: common name, Latin name, density (q/m³ — see `database.md`;
  used to derive per-tree mass for samples and marks), and a "Minore"
  (minor) flag. Minor species are grouped under a single "Altro" entry in
  Prelievi (see `page-prelievi.md`); the flag is editable here.

## Future production (Produzione futura)

Visible to writers. Determines the data set used to compute per-parcel forecast
production ("prelievo previsto") in the Bosco page (see page-bosco.md).

It consists of a pulldown of available harvest plans. Exactly one can be
selected at a time. It defaults to the plan whose (start year, end year) overlap
the current year and whose end year is greatest. Robust in the event that no
harvest plan exists.

Changing this setting invalidates the future_production.json digest.

## Dendrometric data (Parametri dendrometrici)

Visible to writers. Determines the data set used to compute dendrometric
parameters in the Bosco page (see page-bosco.md > Dendrometria).

The top of the section is a line listing the current total number of trees,
regions, and parcels used for the dendrometric parameters.

Below is a multiselect of available surveys (Rilevamenti), identical to the
multiselect in the hypsometry section (see hypsometry.md). The currently
selected surveys are highlighted. Defaults to the first survey by name alpha
order, and is robust to there being no surveys.

Below the multiselect is an "Aggiorna" submit button that causes the user's
selection to go into effect.

Updating this setting invalidates the parcel_dendrometry.json and
parcel_dendrometry_points.json digests used by Bosco.

## Hypsometric parameters (Parametri ipsometrici)

Visible to writers and admins, below the trees section. A read-only table of
the active hypsometric parameter set, with Importa / Esporta / Elimina controls
and a "Calcola nuovi parametri" panel that fits new coefficients from selected
surveys. Fully documented — behavior, the compute→accept flow, the CSV format,
and the served digest — in [`hypsometry.md`](hypsometry.md).

## App users

This section is visible only to admins.

Admins can create and edit users. Table columns: first/last name,
username, email, login method, created-at, active.

Form fields:

- Login method radio (password / OAuth).
- Email (required; for OAuth, must match Entra account).
- Username (password only; auto-populated from email for OAuth). Hidden
  when OAuth is selected.
- Password (repeated; hidden when OAuth is selected).
- First name, last name.
- Role pulldown (Membro / Redattore / Amministratore).
- Active checkbox — only active users can log in.

The initial admin account is configured at server installation time.
OAuth users must be pre-added with a matching email to whitelist them.
