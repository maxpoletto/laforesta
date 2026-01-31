# Particelle

"particelle" is a visualization app for properties of forest parcels.

## Description of forest parcels

Forest parcels are specified in the "Prelievo per particella" sheet of
foresta.xlsx.

Each forest parcel is denoted by a "Compresa" and a "Particella".
The possible values of Compresa are Capistrano, Fabrizia, Serra.
Particella values are short alphanumeric codes like 1, 10b, 9.

Each parcel has the following interesting characteristics initially
(we may add more later):

* Governo
* Area (ha)
* Età media
* No. fustaia
* No. ceduo
* m3/ha nuovo
* incr/ha nuovo


## Description of the app

The app has two modes: map and tabular.

In map mode, the main element is a map that uses deck.gl and maptiler.

The map displays each parcel. At the moment, we do not have GPS coordinates
of the perimeter of the parcels, so we display the parcels as _non-overlapping_
squares whose size is proportional to the value of Area (ha).
* The parcels with Compresa=Serra should be clustered around (38°33'40.7"N 16°18'10.4"E).
* The parcels with Compresa=Fabrizia should be clustered around (38°28'13.2"N 16°16'56.4"E).
* The parcels with Compresa=Capistrano should be clustered around (38°41'55.5"N 16°18'32.3"E)

The map starts out at a zoom level sufficient to clearly view all the Compresa=Serra parcels, but
not necessarily the others.

Hovering over a parcel provides the name (Compresa and Particella) and all the characteristics listed
above (Governo, Area, etc.).

The map has "full-screen" and zoom widgets on the upper left.

On the upper right of the map is a small control panel with a pull-down menu that describes the current
visualization parameter, chosen from among the characteristics listed above. The current visualization
parameter describes how the parcels are colored. For enumerated values (currently only "Governo")
a parameter values is mapped to a different color (e.g., green, blue, yellow). For numeric values the
parameter value is mapped to different shades of the same color, e.g., from very light green for the
minimum value to very dark green for the maximum value.

In addition, the visualization parameter controls a table beneath the map (not visible if the map is
in full screen mode). This table displays three columns: Compresa, Particella, and the visualization
parameter.

The control panel also includes a "summary table" button ("Tavola d'insieme"). Pressing this button
hides the map view and displays a single large table that has Compresa, Particella, and one column
for each of the characteristics described above. It contains a "map" ("Cartina") button (in the same
place on the screen as the summary table button) that toggles back to map view.

The color scheme is pastel, not very saturated. The map layer is topographic.
The implementation is in a single html file that includes all the necessary javascript and fetches
a CSV version of the "Prelievo per particella" sheet.
