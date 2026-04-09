# Abies: integrated management of forestry company operations

Abies is a full-stack web app used to manage production operations of a forestry
company, including forest harvests, sawmill production, biomass energy
generation.

It is intended to help with day to day operations, track production, and monitor
forest health and productivity.

# User base

The target audience is office workers and field staff at a small (40-employee)
lumber company that also produces electricity from biomass and solar. They
currently use Microsoft 365 Business and complain about its slowness. They are
unsophisticated computer users. They employ Word and Excel daily but only use
basic features (e.g., they do not know how to set up Excel pivot tables).
Simplicity and speed are key requirements.

# Priorities

1. Correctness
2. Security
3. Speed (the UI is snappy and works well over 3G)
4. Low maintenance over time

# Functional overview

Version 1 of the app covers the following functional areas, which we call _domains_:

- Forest visualization and planning. Geospatial tool that displays historical
  harvest data, supports planning and execution of forest surveys (setting up
  sampling locations; recording geo-referenced annotations; etc.), and displays
  other geo-located information about the forest, including satellite imagery
  and health metrics.

- Forest harvesting. A log of the daily activities of crews of lumberjacks,
  including how much wood of different species was harvested where and with what
  tractors. The tool also supports day-to-day operations by generating PDF
  record slips that the crews fill out and provide to the office staff for data
  entry.

Future versions will also cover the following domains.

- Sawmill. A log of daily sawmill operations, including volume and
  type of wood products, maintenance activities, failures or other incidents,
  etc., as well as monthly rollups of income and other parameters.

- Biomass plant. A log of daily biomass operations (amount of biomass
  consumed, energy produced, various operational parameters), as well as monthly
  rollups of energy produced, energy consumed, income broken down by various
  parameters, etc.

- Photovoltaic plant. Daily log of production, monthly log of
  verified production and revenue.

- Fuel facility. Log of operations on the company's diesel fuel tank: who
  refueled what vehicle, how much fuel they used, refueling of the fuel tank
  itself, etc.

Each of these domains is handled in a distinct page of the app and is separate
from the others. However, the outputs of some areas are inputs to others (for
example, wood from the forest flows into the sawmill and biomass plant).

Historical data can be displayed in searchable tables and in graphical charts,
as well as in the forest maps.

# Architecture overview

The app is a SPA-lite: Django serves a single shell page after authentication,
and vanilla JS handles client-side routing, data caching, and content rendering.
There is no JS framework.

- Vanilla JS on the client, with minimal dependencies:
  - @maxpoletto/sortable-table for tabular data display
  - Chart.js for data visualization
  - Leaflet for mapping
- Django on the server, served via Apache with mod_wsgi:
  - JSON endpoints for data (consumed by sortable-table, Chart.js, and Leaflet)
  - HTML fragment endpoints for forms (injected into the shell)
  - django-allauth for authentication
  - django-simple-history for audits
- SQLite for storage.

# Security

Data security (both privacy and integrity) is important because the app stores
the core of the company's operations.

## Permission model

We have a three-role permission model, "admin", "writer", and "reader". Readers
have read-only access. Writers can modify any data not related to access
control, performing insertions/edits/deletes on any tables that allow it (see
details below). "admins" can also create and edit users.

If using username/password pairs, users can change their own password.

## Authorization

The app is access-controlled on the server side.

Authorization supports MS 365 credentials and user/password pairs via
django-allauth,
[integrated](https://django-axes.readthedocs.io/en/latest/6_integration.html#integration-with-django-allauth)
with django-axes for basic brute-force protection.

OAuth must be pre-provisioned (the email address whitelisted) by an admin.

Session expiration is server-configurable and defaults to 12 hours.

In the future we may need to support other OAuth identity providers.

## Auditing

The app records all writes using django-simple-history.

The audit log is readable and searchable by all users. (More on this below in
the UI section.)
 
# UI architecture

The app is structured as a SPA-lite. After authentication, Django renders a
single shell page that persists for the duration of the session. All subsequent
navigation happens client-side without full page reloads.

## Shell and header

The shell is a single Django template containing just:
- The header, shared by all domain pages.
- A content area where the domain-specific page content is rendered.

The shell is rendered once and never reloads during normal use.

The header is adaptive for desktop and mobile. On narrow displays it contains only:
- The log of the company.
- The name of the currently active domain (Forest, Harvesting, Sawmill, Biomass,
  Photovoltaic, Fuel, Audit, Settings)
- A hamburger icon for a menu that allows switching to other domains.

On wider displays it contains:
- The logo and name of the company.
- The names of the domains as tabs, with the currently active domain highlighted.
- No hamburger icon.

The header is fixed in the viewport. Content scrolls beneath it.

## Routing and URLs

URLs are human readable and are the canonical representation of the view state.
They encode the domain (in the path), and any data filters, chart selection,
etc. as query parameters.

Changing domain or page-specific parameters changes the URL via
`history.pushState()` and renders the appropriate content. The back button works
via `popstate`.

All URLs are bookmarkable. A user can send a URL to a colleague and it
reproduces what they were looking at.

Because of this design choice and the relatively flat tabbed navigation
structure, we omit any explicit "breadcrumb" navigation component.

The paths and options for each domain are specified in the detailed descriptions
below.

## Data storage and serving

All data except some geographic data (e.g., satellite images) lives in
relational form in SQLite on the server.

However, to reduce latency, it is always served as compressed pre-computed JSON.
This JSON representation is generated server-side whenever a related SQLite
table changes. The mapping of relational tables to JSON files is specified in
the detailed domain descriptions.

For tabular data, the format is an array of arrays. The first entry denotes
table headers and subsequent entries are pure data. Every entry has a hidden
"row_id" field, used for updates (see below).

## Caching

Fetched data is cached client-side in memory. The cache is keyed by a "data_id"
that identifies a particular dataset corresponding to a server-side JSON file,
e.g., "daily harvest operations" or "monthly sawmill production", and it stores
the last refresh time.

The maximum size of the cache is the sum of all tabular and pre-processed data,
on the order of 10-20 MB uncompressed, plus 10-100 MB of satellite imagery in
the worst case.

Page reloads clear all cache state.

When changing domain pages:

1. The app renders from cache immediately if data is available.

1. The app also sends background conditional GETs to the server with
   If-Modified-Since for every dataset displayed in the domain page. If the
   server returns 304 Not Modified, no action is required. If it returns 200 OK,
   the app updates the cache with the newly received data and re-renders. In
   both cases it updates the dataset's last refresh time.

Conditional GETs also run once every 5 minutes (on a timer) for every dataset
visible in a particular domain page (i.e., the page will refresh even if the
user performs no navigation).

Domain switching feels instant for previously-viewed data.

## Data entry and cache updates

Data entry forms are Django-rendered HTML fetched as fragments into the shell's
content area.

Each form has custom HTML and validation JS as needed, but common patterns (form
interception, error display) are extracted into shared libraries.

Fields that are enum-like (correspond to finite sets of values defined within
the app itself) are implemented as pull-downs. These include worker names, crew
names, tractor names, and tree species names (see below for details).

The process of data entry runs as follows:

1. The user initiates a data addition or edit by clicking on a UI button (the
   visual details of this are below).
1. JS fetches the form HTML from Django (including the CSRF token).
1. The form is rendered in the current page (it replaces the current view). The
   URL *does not change*. This is the one exception to the "canonical
   representation of view state" rule, since we never need to share the input
   form.
1. Client-side JS validation provides immediate feedback: the submit button is
   inactive until JS validation passes.
1. On submit, JS intercepts and forwards the POST request (including the CSRF
   token, present in the "csrfmiddlewaretoken" hidden field), and waits for a
   response.
1. The server provides authoritative validation.

The server response has one of three values. The payload is always JSON.

1. Success: Code = 200 OK, payload = { data_id: X, row_id: Y, "record": [
   Y, ... ] }

   The client updates row Y of entry X in the case with the new record and
   refreshes the tabular display.

   Note that a future background conditional GET might refresh other cache
   entries, such as those corresponding to digested data for graphs. Concrete
   example: user enters data corresponding to a new harvest operation. Tabular
   of harvest operations updates immediately. Digested data for bar chart of
   monthly harvests might be loaded after the next conditional GET.

1. Validation error: Code = 400 Bad request, payload = { status:
   "validation_error", message: "...", html: "..." }

   The page displays the error message in a modal and again displays the blank
   HTML form given by the "html" field.

   This rarely happens, since most server-side validation is consistent with
   client-side validation.

1. Conflict: Code = 400 Bad request, payload = { status: "conflict", message:
   "...", data_id: X, row_id: Y, "record": [ Y, ... ], html: "..."}

   This error happens on attempted edit or delete. Another user edited or
   deleted the entry between the time when the current page's last cache refresh
   and the time of submit. The HTML contains the form populated with the current
   server state. If the record has been deleted on the server, the message
   provides this information and the user can click "submit" to re-add the
   record. If the user escapes out of the edit form, the cached data is updated
   with the returned record, as for a successful update.

## Data deletion and cache updates

If the user deletes a record (more on the UI details of this below), the UI
displays an alert warning that the action cannot be undone. 

If the user confirms, a POST is sent to the server as for data insertion / edit
above.

1. Successful responses contain a row_id but no record field. The client removes
   the given id from the cache.

1. No validation errors are possible.

1. Conflict means that a row was edited since last cache refresh. The response
   contains no HTML but a valid record. The cache is updated as for a successful
   update. The error message is displayed in a modal and the user has the chance
   to try deletion again.

## Error reporting

Errors are reported in modals. Errors include validation errors, conflicts, and
other conditions such as network errors.

An error modal contains the error message and a dismiss button. It can also be
dismissed with the "escape" key.

## CSS

No style is defined in-line in HTML. All style is defined in CSS files. 

Most styles are maintained in a single common.css style file. Styles that are
only required on one page appear in a page-specific style file.

## Javascript dependencies

All JS dependencies (Leaflet, Chart.js, sortable-table, (Google) fonts, etc.)
are minified and vendored, served from Django's static/vendor. There are no
external dependencies at runtime. A Makefile target (`make update-vendor`)
re-copies from source when needed.

# UI design patterns

## Objectives

The objectives of the visual design are:

- Readability: data is presented simply and clearly, with good use of screen
  real estate on both desktop and mobile.
- Predictability: consistent visual guidelines, no unexpected behavior.
- Discoverability: navigation is easy and fast.
- Restfulness: cognitive and visual load are low.

## Fonts and colors

DM Sans is used throughout.

The UI is strictly two-dimensional: there are no drop-shadows, text inputs are
flat, scroll bars are flat.

Page margins are moderate (15 px) on desktop and almost disappear (2 px) on mobile.

Text inputs have very slightly rounded corners (2-4 px radius).

Buttons have rounded corners (4-8px radius). They are dark green and turn
lighter when hovered over.

Horizontal rules outline the page header as well as collapsible elements  (more
on these in "Detailed description" below). They are thin (4px), dark green, and
rectangular.

## Data display overview

Within each domain page, information is displayed consistently in one of three
ways:
- Tabular data is consistently displayed in sortable-table (more on this below).
- Graphical data is displayed using Chart.js.
- Geographic data is displayed using Leaflet.

Some visual elements may be hidden within collapsible sections.

## Tabular data

All tabular data appears in sortable-tables.

- All fields are sortable.
- All tables are searchable via a text input immediately above the table, on the
  left. Search is immediate (no search button) after a 1/2-second sleep to
  debounce rapid keyboard input.
  - Conceptually (not necessarily in actual implementation), the search operates
    as follows:
    1. Split the search text on whitespace.
    1. For every row, join all fields into a single string.
    1. Return all rows that contain all elements of the search text in the given
        order.
  - The search acts purely as a filter. The table does not move, but displays only
    matching rows. Any pre-existing sort order is preserved.
- A table displays rows as far down as the bottom of the viewport. If there are
  more rows, the table has a scrollbar that is separate from the page scrollbar.
  (On mobile, there is enough lateral space to allow the user to also scroll the
  page, not just the table).
- Tables have 1px medium-grey borders and column headers have light grey background.

Additionally, for users with role "writer" or "admin", tables may (depending on their semantics, see detailed descriptions below) allow modification:

- Tables that allow row addition have a "+" button below the bottom row, on the
  right.
- Tables that allow row editing have a "pencil" icon on the right of each row.
- Tables that allow row deletion also have a "garbage can" icon on the right.

## Graphs and charts

All graphs and charts are implemented in Chart.js.

All charts have y-axes that begin at 0.

All color maps range from yellow-green (for low values) to dark green (for high values).

Graphs occupy the full screen width and legends appear below the graph (on both
desktop and mobile).

## Maps

Maps have the following visual structure:

- A navigation bar on the right.
- The following Leaflet tools appear in the upper left corner, top to bottom:
  - A hamburger button to hide/display the navigation bar.
  - A location pin.
  - Zoom +/- buttons
  - A ruler

The top of the navigation bar always contains, top to bottom:
- A status panel
- A map type selector. Buttons for OSM, Topo, Satellite.
- A pull-down region ("compresa") selector. Choosing a region centers it on the
  map and sets the zoom level to the most detailed level that still includes the
  full region in the viewport.

Below these elements are application-specific controls, organized in collapsible
sections.

This structure is very similar to that of the Boscoscopio app (laforesta/bosco/b).

## Modals

Modals have a consistent style, with slightly rounded corners and thin dark
green borders. Their background is white but they cause the rest of the page to
darken by about 50%.

They are used to display error message (with red text) and  help information
where available (e.g., "?" links next to map navbar elements).

## Accessibility considerations

In its initial version, given the target staff, Abies has no special
accessibility features (high contrast, etc.), though of course enlarging fonts
in the browser is always an option for users.

# Storage

As mentioned previously, core relational data is stored in SQLite on the server.

SQLite is appropriate because:
  - The app has a small user base, and is read-mostly.
  - WAL mode handles concurrent reads and writes fine at this scale.
  - No moving parts: no database server process to maintain.
  - Simple backups (copy one file).
  - Django's ORM abstracts the DB, so migrating to Postgres later is
    straightforward.

In addition, summary statistics are computed periodically and stored as
compressed JSON files on the server filesystem for consumption by visualization
tools (charts and maps). (The files are compressed at creation time to decrease
load during serving.)

Satellite imagery (GeoTIFF) and GeoJSON geometries (for geo data) are also
stored as files on disk.

## Concurrency

We handle concurrent modification of database data with row-level optimistic
locking. Every table has an implicit version column (not listed in the schemas
below) that increments on every save. The edit includes the version as a hidden
field; the server checks the version on POST, and if the version has changed
that indicates a write conflict.

## Disconnected operation

The app does not support disconnected operation but works well on relatively
low-bandwidth (e.g., 3G) connections thanks to pervasive caching, pagination of
relational data, and compressed, pre-processed data files.

## Backups

A cron job on the server runs a nightly backup of the database.

Compressed nightly backups are retained for a month, and one copy a month
(arbitrarily on the first day of the month) is uploaded to OneDrive using the
Microsoft Graph API.

Setting up Microsoft Graph OAuth credentials is part of the initial
configuration flow of the server app (in the shell, not via a web UI).

## Data import

Importing data is done through single-use ETL scripts that parse existing CSV
files and normalize them.

## Data export

Every domain page includes a button (upper right) that allows its data to be
exported as a CSV file.

## Database model

Here we describe only common tables used across the app. Per-domain tables
appear in the detailed app description below.

- region: (id:int, name:string)
  - Denotes a forest region or "compresa".

- eclass: (id:int, name:string, coppice:bool, min_harvest_volume:int)
  - Represents a parcel economic class. It may be coppice or high forest
    (coppice=false).
  - Characterized by a minimum volume (m3/ha) before harvesting is permitted.

- harvest_plan: (id:int, description:text, interval:int)
  - Represents a harvest plan for a parcel.
  - 'interval' denotes harvest interval for coppice parcels.

- parcel: (id:int, name:string, region_id:int, class_id:int, area_ha:int,
  year:int, location_name:string, altitude_min_m:int, altitude_max_m:int,
  aspect:str, grade_pct:int, desc_veg:string, desc_geo:string,
  harvest_plan_id:int)
  - Represents a forest parcel. 'name' is typically an alphanumeric string like
    '11' or '2a'.
  - 'area_ha' is surface area in hectares.
  - 'ave_age' is the average age of trees in the parcel.
  - altitudes are in meters.
  - 'desc_veg' and 'desc_geo' are strings that describe the vegetative and
    geologic state of the parcel, respectively.

- tractor: (id:int, manufacturer:string, model:string, year:int, active:bool)
  - Represents a tractor. 'year' denotes date of manufacture.
  - Retired tractors have active=false.

- worker: (id:int, first_name:string, last_name:string, year:int, notes:string, active:bool)
  - Represents a worker, e.g., a lumberjack.
  - 'year' denotes birth year and is optional.
  - Workers who are no longer employed have active=false.

- crew: (id:int, name:string, active:bool)
  - Represents a team of workers, e.g., a group of lumberjacks.

- staffing: (crew_id:int, worker_id:int)
  - Maps workers to crews. A worker may appear in at most one crew at a time, so
    worker_ids in this table are distinct.
  - A crew need not have staffing for it to be used, e.g., in harvest tables.
    Staffing details are optional.

- `species`: (id:int, common_name:string, latin_name:string, active:bool)
  - Represents a tree species.

# Internationalization

The app initially only supports Italian. The URL paths are in Italian also.

But there are no inline strings in the code. All are named constants (in both
Python on the backend and JS on the front-end) to make a future
internationalization easier. Path names are also named constants. (The
assumption is that any future language choice will be at the level of the entire
app, not individual users.)

Numbers and dates are represented using Italian locale.

Exported CSV uses semi-colons as separators.

The _code_ itself (variable names, function names, etc.) is all in English. We
use terms like 'coppice' in the code instead of 'ceduo', and so on.

# Mobile

The app is usable on mobile in portrait mode (without needing to switch to landscape).

More on this is in the detailed description below.

# Django project structure

The Django project is organized into apps by domain.

- `core` includes common models (see above), the shell template, and common
  CSS/JS.
- Then there are apps for each domain: 'forest', 'harvest', 'biomass', 'pv',
  'fuel', etc., that contain models and templates for each domain.

# Code location, deployment, and releases

The Django project is rooted at laforesta/abies in the code repository, and is
served from a similar location once deployed (i.e.,
https://laforesta.it/abies/).

Abies is designed to be deployed on a single server (e.g., laforesta.it).

Official releases are numbered using git tag.

Abies is deployed via a Docker image that includes all dependencies, and it
mounts the host filesystem for:

- SQLite database file;
- Pre-processed JSON files;
- Backups.

# Testing

Python tests use pytest-django with pytest-cov for coverage reporting
(--cov-report=term-missing), matching the existing pdg-2026 test setup. Tests
cover models, views, form validation, and ETL scripts.

Client-side JS tests use the existing `node tests.js` pattern (see bosco/b/).

The test instance of Abies is deployed locally and does not use Docker, to speed
up testing and debugging.

# Relationship to existing "bosco" apps

The Forest Visualization domain of Abies subsumes "boscoscopio" and certain
other "bosco" apps. The "bosco" apps remain unchanged for now but will
eventually be taken offline and replaced entirely by Abies.

# Detailed description

## Login page

The login page is mostly blank. In the center, inside a dark-green-bordered
square, is the company logo and name, entry fields for username and password,
and a "Log in with Microsoft button".

Upon successful login, users land on the Forest Visualization page.

## Settings page

The settings page contains several collapsible sections separated by horizontal
rules. All sections are collapsed by default. Not all sections are visible to
all users (details below): if a section is not visible, it is completely hidden,
not just collapsed.

### Personal settings

This section is visible to all users (reader, writer, admin) who use password
authentication. It provides two simple text-entry fields, "new password" and
"repeat new password". They must of course match.

### Workers, crews, tractors, and trees

This section is visible only to writers.

They can create and edit workers, crews, tractors, and tree species.

Each of these entities is configured in its own collapsible section.

Each section contains a corresponding sortable table.

Each of these sortable tables supports adding and editing entities, but not
removing them.

In each table the rightmost column is titled "active" and denotes whether the
entity (worker, tractor, etc.) should appear as an option in new input forms.

Above each table, on the right of the search box, is a checkbox for "Only
active". It is checked by default to avoid clutter.

The tables differ in the columns that they display (and therefore the data entry
fields that the corresponding input modal provides):

- Workers: last name, first name, birth year (optional), crew name (optional), notes (optional).

- Crews: name, notes (optional).

- Tractors: manufacturer, model, year.

- Trees: common name, Latin name.

### App users

This section is visible only to admins.

Admins can create new app users and edit existing users.

The sortable-table contains the following columns:

- First and last name.
- Username or OAuth identifier.
- Login method (one of password or OAuth).
- Created-at time.
- Active status.

Users are editable ("pencil" icon next to each row) and creatable ("plus" icon
at bottom of table).

The user input/edit form has the following fields:

- Login method radio button (password or OAuth).
- Username text input (or expected email address for OAuth).
- Password (repeated text input, values must match). Only visible if login
  method is password.
- Role (pull-down menu with three choices (reader/writer/admin)).
- Active status (checkbox). Only active users can log in.

Changes take place when the admin presses the "Submit" button.

The initial admin account is configured at server installation time.

The admin must add an OAuth user in order for the email address to be
whitelisted for OAuth access.

## Audit page

This page is visible to all users.

The audit page displays a sortable-table table with the following columns:

- time and date, user, table name, action (insert/edit/delete), value before, value after

This information comes from django-simple-history. The table is not editable,
but it is searchable and sortable like all other sortable-tables.

## Forest visualization

TBD.

## Harvesting

### Data model

- operation_type: enum (tronchi, cippato, ramaglia, pertiche_puntelli, pertiche_tronchi)
- note: enum (none, PSR, fitosanitario, catastrofato)
- harvest: (id, date, operation_type, parcel_id, team_id, vdp (int, nullable),
  prot (int, nullable), quintals (float), note (enum), extra_note (text),
  version (int), created_at, modified_at)
- harvest_species: (harvest_id, species_id, percent) — PK is (harvest_id, species_id)
- harvest_tractor: (harvest_id, tractor_id, percent) — PK is (harvest_id, tractor_id)

Note that "Coppice" (ceduo) is both a stored property of parcels and a computed
property of individual harvest operations (true if castagno > 50%). It is
possible to extract some coppices from a parcel that is otherwise high forest
(fustaia).

### UI

- "Aggiungi dati", which sends us to the data entry page.
- "Modifica dati" displays a list of all the entries and allows the user to
  select one to edit (then goes to the data entry page).
- "Visualizza dati", which for now just lists the raw data.

Both "Modifica" and "Visualizza" show filters according to the following criteria:
- Date (a double-ended slider with year granularity, see bosco/a/range-slider.*).
- Team
- Tractor
- Region
- Parcel within region
- Species (any matching species)
- Note value (see Data model for valid enum values)

The data is displayed in a sortable-table.

In "Modifica" mode, each row has an "edit" link that navigates to the edit form
for that row (prepopulated with existing values). On successful save (or
cancel), the server redirects back to the list with filters preserved via query
parameters. On a write conflict (version mismatch), the form re-renders with
the current data from the database and an error message.

For mannesi we have entry fields for date, team name, type of operation, amount
of material harvested, percentage breakdown by species and by tractor, etc.

Some fields (e.g., team names) come from a finite predefined list that can be
configured via the Settings section (see above). These
fields should be implemented as pull-downs, such that tabbing over to them and
starting to type also selects the right row in the pull-down.

Some fields (species and tractors) can have multiple values that must add to
100% (e.g., harvest operation was 80% fir, 20% pine, using 50% tractor A and 50%
tractor B). For these fields, we list all values and provide next to each one
both a numerical entry text input and a "100%" button for the common case.

Fir: [box] [100%] Beech: [box] [100%] ...

Pressing a button sets the corresponding box to 100 and the others to 0.

We should also consider other ways to make the data entry faster, e.g.,
prepopulating the date field with the current date or the most recently entered
date.

Note, because of "modifica dati", data entry forms work both in create and edit
modes. In edit mode, the form is prepopulated with existing values. We detect
stale edits as described in storage below.


## Biomass power plant

TBD.

## Photovoltaic plant

TBD.

## Fuel facility

TBD.
