# Mannesi page

This page appears to the right of Prelievi. It provides features to manage
lumberjacks.

Three foldable sections, described below.

For database schema, see database.md > Mannesi data.

## 1. Ore (Work time tracker)

A sortable-table with the standard Search box and "Esporta CSV" button on top, and the following columns:

Data, Squadra, Ore, Note

Default sorted in reverse chronological order. Like everywhere else, Date is in YYYY-MM-DD format.
On the right writers also see the standard pencil and garbage can tools: behavior identical to Prelievi and other tables.

Below the table, on the right, a green "+ Aggiungi" button.

Clicking on the Aggiungi or pencil buttons opens an "Acconto" modal with:
Date
Pull-down with active teams (squadre)
Hours text input (can be fractional)
Note text input

Hours must be > 0, team and date cannot be null, else showFormError.

## 2. Acconti (Production credit tracker)

Essentially identical to Ore above, with "Quintali" in place of "Ore".

## 3. Ricevute

Very simple appearance:
- A month picker widget (e.g., "gennaio 2026"), and a green "Genera" button.

On Genera, browser downloads a PDF file ("ricevute-mannesi-YYYY-MM.pdf") that
contains a report for every team that has at least one harvest row for the
desired month.

The reports appear in alpha order by team. Each report is one or more A4 pages
and begins on a new page.

The report contains:

Squadra [name]
Mese [month-name YYYY]

Ore lavorate:       [sum over ore (above) in YYYY-MM]

Riassunto           Quintali
[Product 1]         [sum over product 1 quintals in YYYY-MM]
...
[Product N]         [sum over product N quintals]
Totale produzione   [sum over the rows above]

Acconti             [sum over acconti (above) in YYYY-MM]

Totale              [totale produzione - acconti ]

Dettaglio produzione
[Excerpt of the harvest table, limited to the team and YYYY-MM]
Columns: Data, Compresa, Particella, VDP, Tipo, Q.li, Note, [% for every major species]
