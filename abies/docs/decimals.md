# Desired end state (goal)

User-visible numeric locale behavior:

1. We do not display or recognize the thousands separators at all.
2. The only numeric locale setting is whether decimal separator is "," (e.g.,
   locale=it_it) or "." (e.g., locale=en_us).
3. The choice of decimal separator is performed (like with text strings) at
   backend setup time (e.g., it is either an "Italian" abies install or a "US"
   abies install).
4. *ALL* user-visible numbers that have a fractional component are displayed
   with the desired separator. No exceptions.
5. We are lenient about input: a user could input "," or "." as decimal
   separator, and we accept it.
6. Search in sortable-table works as the user would expect: i.e., searching for
   "3,14" in an Abies table with "," decimal separator returns lines that (in
   the backing array) contain the numeric value [3.14].
7. It is not permitted to enter 0 as a tree density, diameter, or height (UI
   displays form error).
8. CSV output is driven by install-time locale (',' field separator / '.'
   decimal separator) or (';' field separator / ',' decimal separator). All
   dates are ISO8601 YYYY-MM-DD format.
9. CSV input is lenient and allows both (',' field separator / '.' decimal
   separator) and (';' field separator / ',' decimal separator).

Implementation features:

1. Data on the backend (in the DB and in digests) is stored in dotted decimal
   format (3.14).
2. Python:
   * All number are parsed as Decimals. `grep parse_float` returns empty.
   * Decimals are passed to math.* functions (e.g., log) for the purpose of
     regression computations.
   * We return None / raise an error on division by zero, undeflow, overflow,
     and any other conditions that may return NaN or ±Infinity.
3. Javascript:
   * Integer inputs are parsed using parseInt, not parseFloat.
   * There is no use of parseFloat. `grep parseFloat` returns empty.
   * All parsing and formatting happens through parseInt or functions centrally
     defined in base/static/base/js/format.js (including wrappers around
     Intl.NumberFormat). There are no other page-specific parsing or formatting
     functions.

# Inventory

I believe this is comprehensive, but of course I may have missed something.
(Bosco/Forestscope is unimplemented and out of scope.  Individual CSV
import/export columns are not listed — CSV is covered wholesale by §8–9.)

* Integer output fields
  * Piano di taglio
    * Calendar table
      * Anno previsto
      * Anno effettivo
      * Turno
    * Martellata (marks) table
      * Numero
      * D(cm)
    * Item-view metadata panel
      * Anno previsto, Anno effettivo, Turno
  * Campionamenti
    * Grid sample area detail popover
      * Numero
      * Raggio (m)
      * Alt. (m)
    * Sampled-trees table
      * N. albero
      * Pollone
      * D(cm)
      * L10(mm)
    * Auto-grid preview
      * Punti
    * Grid summary
      * N. aree
      * N. rilevamenti
    * Survey summary
      * N. aree visitate / totali
    * Map tooltips (griglie / rilevamenti)
      * adc N.
      * N. alberi
  * Prelievi
    * Harvests table
      * VDP
  * Impostazioni
    * Hypso table
      * n

* Fractional (decimal) output fields
  * Piano di taglio
    * Calendar table
      * Volume previsto (m³)
      * Volume martellato (m³)
      * Volume effettivo (m³)
      * Superficie intervento (ha)
      * Superficie totale (ha)
    * Martellata (marks) table
      * h(m)
      * V(m³)
      * m(q)
      * Lat, lon
    * Martellata totals
      * Volume totale
      * Massa totale
    * Item-view metadata panel
      * Volume previsto (m³) / martellato (m³) / effettivo (m³), Superficie intervento / totale (ha)
    * Tree-mark form live preview
      * V(m³), m(q)
  * Campionamenti
    * Grid sample area detail popover
      * Lat, lon
    * Sampled-trees table
      * h(m)
      * V(m³)
      * m(q)
      * Lat, lon
    * Auto-grid preview
      * Area singola adc (m²)
      * Superficie totale (ha)
    * Tree form live preview
      * V(m³), m(q)
  * Prelievi
    * Harvest table
      * Q.li
      * Volume
      * All species and tractor columns (quintals)
  * Impostazioni
    * Specie table
      * Density
    * Hypso table
      * a, b, r^2

* Integer input fields
  * Piano di taglio
    * Nuovo/modifica piano
      * Anno inizio
      * Anno fine
    * Nuovo/modifica intervento
      * Anno previsto
    * Nuovo albero martellato
      * Numero
      * D(cm)
  * Campionamenti
    * Nuova griglia (auto)
      * Raggio (m)
    * Nuova/modifica area
      * Raggio (m)
      * Alt. (m)
    * Nuovo/modifica albero
      * D(cm)
      * L10(mm)
      * Coppice per-shoot: D(cm)
  * Prelievi
    * Nuovo/modifica prelievo
      * VDP
      * Species % (per species)
      * Tractor % (per tractor)
    * Year-range filter slider
      * Anni
  * Impostazioni
    * Trattore edit
      * Anno
    * Hypso compute
      * N minimo

* Fractional (decimal) input fields
  * Piano di taglio
    * Nuovo/modifica intervento
      * Volume previsto (m³)
      * Superficie intervento (ha)
    * Nuovo albero martellato
      * h(m)
      * Lat/lon
  * Campionamenti
    * Nuova griglia (auto)
      * Copertura (%)
    * Nuova/modifica area
      * Lat, lon
    * Nuovo/modifica albero
      * h(m)
      * Lat/lon
      * Coppice per-shoot: h(m)
  * Impostazioni > Specie edit
    * Density

* Special surface (decide whether to localize)
  * Controllo (audit): old/new value columns render numeric field values
    (VDP, Q.li, Volume, Density, Anno) as composed "label: value" strings,
    formatted server-side.  Historical diff values — lower priority.
