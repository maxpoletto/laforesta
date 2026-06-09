# Mannesi page

This page appears to the left of Prelievi. It provides features to manage
lumberjacks.

Three foldable sections, described below.

## Verbali di pesata

Supports generating a PDF with record slips that lumberjacks use for reporting harvests.

### Section layout

Text inputs:

- Numero iniziale (defaults to (max(VDP)+1) from Prelievi page)
- Targa autocarro (license plate, persistent after write, even across reloads / logouts)
- Numero di verbali (number input, accepts multiples of 4)

Below, in green, "Genera" button.

When Genera is pressed, browser downloads a PDF file ("vdp.pdf") with the report slips.

### PDF format

Slips are printed 4-to-a-page on a 2x2 grid on A4 paper, portrait mode.


Each verbale contains:

Data _____________   N. [VDP number]

Compresa [Compresa 1] ... [Compresa N]
Particella [ blank  ]

[Product 1] [Product 2] ... [Product N]
Targa [license plate value]

Essenza             %
[Major species 1]
...
[Major species N]

Peso lordo ql [           ]
Tara       ql [           ]
Peso netto ql [           ]

Squadra  [         ]

Firma    ___________


## Acconti

A sortable-table with the standard Search box and "Esporta CSV" button on top, and the following columns:

Data, Squadra, Quintali, Note

Default sorted in reverse chronological order. Like everywhere else, Date is in YYYY-MM-DD format.
On the right writers also see the standard pencil and garbage can tools: behavior identical to Prelievi and other tables.

Below the table, on the right, a green "+ Aggiungi" button.

Clicking on the Aggiungi or pencil buttons opens an "Acconto" modal with:
Date
Pull-down with active teams (squadre)
Quintals text input
Note text input

Quintals must be > 0, team and date cannot be null, else showFormError.


## Ricevute

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

Riassunto     Quintali
[Product 1]   [sum over product 1 quintals in YYYY-MM]
...
[Product N]   [sum over product N quintals]

Acconti       [sum over acconti in YYYY-MM]

Totale        [ ]
