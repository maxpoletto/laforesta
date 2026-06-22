# Squadre page

This page appears to the right of Prelievi. It manages harvesting teams, their
worked hours, production advances, and monthly reports.

Four foldable sections are available. `Personale` is closed by default; `Ore`,
`Acconti`, and `Rendiconti` are open by default.

For database schema, see database.md > Squadre data.

## 1. Personale

Visible to all authenticated users. The table has the standard Search box, CSV
export, and these columns:

Nome, Note, Attivo

A `Solo attivi` checkbox filters out inactive teams by default. Writers also see
the standard `+ Aggiungi` and edit affordances. The modal form creates or updates
`Crew` rows with name, notes, and active flag. Crews are not deleted; they are
deactivated.

API endpoints live under `/api/squadre/crews/`. The data endpoint returns the
shared `crews` digest shape to readers and writers; form and save endpoints are
writer-only.

## 2. Ore

A sortable table with the standard Search box and `Esporta CSV` button on top,
and the following columns:

Data, Squadra, Ore, Note

Default sorted in reverse chronological order. Dates are in YYYY-MM-DD format.
Writers also see the standard edit/delete tools.

The add/edit modal contains:

- Date
- Active team selector
- Hours text input, fractional values allowed
- Note text input

Hours must be greater than zero, and team/date are required.

## 3. Acconti

Same table and modal behavior as Ore, with `Quintali` in place of `Ore`.

The report balance treatment is not a plain same-month subtraction: for a
report month M, the acconto balance is `acconti in M - acconti in M-1`.
Equivalently, an acconto created in month N is added to the quintali balance in
month N and subtracted in month N+1.

## 4. Rendiconti

This section contains a month picker, for example `gennaio 2026`, and a `Genera`
button.

On `Genera`, the browser downloads `rendiconti-squadre-YYYY-MM.pdf`. The PDF
contains one report for every team with at least one harvest row in the selected
month. Reports appear in team-name order. Each team report begins on a new A4
landscape page and may continue onto following pages.

Each report contains:

Squadra [name]
Mese [month-name YYYY]

Ore lavorate:       [sum over Ore in YYYY-MM]

Produzione          Quintali
[Product 1]         [sum over product 1 quintals in YYYY-MM]
...
[Product N]         [sum over product N quintals]
Totale produzione   [sum over the rows above]

Acconti             [monthly acconto balance]

Totale              [total production adjusted by the acconto balance]

Dettaglio produzione
[Excerpt of the harvest table, limited to the team and YYYY-MM]
Columns: Data, Compresa, Particella, VDP, Tipo, Q.li, Note, [% for every major species]
