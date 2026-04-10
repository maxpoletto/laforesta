# Abies: integrated management of forestry company operations

<!-- TOC (line numbers are approximate — re-run grep '^#{1,2} ' to refresh)
  User base .................. 25
  Priorities ................. 34
  Functional overview ........ 41
  Glossary ................... 82
  Architecture overview ...... 98
  Security .................. 115
  UI architecture ........... 168
  UI design patterns ........ 409
  Storage ................... 528
  Internationalization ...... 689
  Mobile .................... 706
  Project structure ......... 712
  Code location / deployment  794
  Testing ................... 811
  Development environment ... 825
  Relationship to bosco apps  846
  Build order ............... 852
  Detailed description ...... 901
    Login page .............. 907
    Bosco page .............. 915
    Prelievi page ........... 1033
    Audit page .............. 1095
    Settings page ........... 1108
-->

Abies is a full-stack web app used to manage production operations of a forestry
company, including forest harvests, sawmill production, biomass energy
generation.

It is intended to help with day to day operations, track production, and monitor
forest health and productivity.

# User base

The target audience is office workers and field staff at a small (40-employee)
lumber company in Italy that also produces electricity from biomass and solar.
They currently use Microsoft 365 Business and complain about its slowness. They
are unsophisticated computer users. They employ Word and Excel daily but only
use basic features (e.g., they do not know how to set up Excel pivot tables).
Simplicity and speed are key requirements.

# Priorities

1. Correctness
2. Security
3. Speed (the UI is snappy and works well over 3G)
4. Low maintenance over time

# Functional overview

Version 1 of the app covers the following functional areas, which we call _domains_:

- "Bosco": forest visualization and planning. Geospatial tool that displays
  historical harvest data, supports planning and execution of forest surveys
  (setting up sampling locations; recording geo-referenced annotations; etc.),
  and displays other geo-located information about the forest, including
  satellite imagery and health metrics.

- "Prelievi": forest harvesting. A log of the daily activities of crews of
  lumberjacks, including how much wood of different species was harvested where
  and with what tractors. The tool also supports day-to-day operations by
  generating PDF record slips that the crews fill out and provide to the office
  staff for data entry.

Future versions may also cover the following domains.

- "Segheria": sawmill. A log of daily sawmill operations, including volume and
  type of wood products, maintenance activities, failures or other incidents,
  etc., as well as monthly rollups of income and other parameters.

- "Biomassa": biomass plant. A log of daily biomass operations (amount of
  biomass consumed, energy produced, various operational parameters), as well as
  monthly rollups of energy produced, energy consumed, income broken down by
  various parameters, etc.

- "Fotovoltaico": photovoltaic plant. Daily log of production, monthly log of
  verified production and revenue.

- "Rifornimenti": fuel facility. Log of operations on the company's diesel fuel
  tank: who refueled what vehicle, how much fuel they used, refueling of the
  fuel tank itself, etc.

Each of these domains is handled in a distinct page of the app and is separate
from the others. However, the outputs of some areas are inputs to others (for
example, wood from the forest flows into the sawmill and biomass plant).

Historical data can be displayed in searchable tables and in graphical charts,
as well as in the forest maps.

# Glossary

- Particella: a parcel of forest land. Usually but not always contiguous. Atomic
  unit for harvest planning.
- Compresa: a forest region comprising multiple particelle.
- Mannesi: a lumberjack. Existing / legacy harvest table is 'mannesi.{xlsx,csv}'.
- VDP and Prot: "verbale di pesata" and "protocollo", id numbers of lumberjack
  harvest operations in mannesi file.
- Bosco apps: Current web-apps in laforesta/bosco. Includes bosco/ads (aree di
  saggio, sample areas), bosco/pai (piante ad accrescimento indefinito, trees to
  be preserved), boscoscopio (geo-based stats).
- PSR: Programma di Sviluppo Rurale. Funding plan for some harvest operations.
- Fitosanitario: a harvest operation performed for forest disease containment.
- Catastrofate: a harvest operation performed to remove trees damaged by
  weather events.

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

## Password policy

Django's default `AUTH_PASSWORD_VALIDATORS` are enabled: minimum length (8
characters), common password check, numeric-only check, similarity to username.

## Rate limiting

django-axes handles login brute-force protection. Data entry endpoints are
rate-limited per user (e.g., 60 requests/minute) to guard against runaway
scripts or bugs.

## Content security

The shell page sets a Content-Security-Policy header that blocks inline scripts.
All user-provided content (notes, descriptions, etc.) rendered into the DOM must
use `textContent`, never `innerHTML`, to prevent XSS.

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
- The logo of the company.
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
The mapping of relational tables to JSON files is specified in the detailed
domain descriptions.

For tabular data, the format is a JSON object with two fields:

    { "columns": ["row_id", "Data", "Compresa", ...],
      "rows": [[1, "2024-03-15", "Alpe", ...], ...] }

`columns` lists column names once; `rows` is an array of arrays containing pure
data. Every row's first element is `row_id`, used for updates (see below). The
client maps columns by name, so adding a column does not break cached data.

### JSON digest regeneration

Digests are regenerated lazily on read, not on write. This avoids wasted work
during batch data entry (where the inserting user's view is already kept current
via cache sync from the POST response).

A small table tracks staleness:

    digest_status(name TEXT PRIMARY KEY, stale BOOLEAN DEFAULT FALSE)

Write path: after a successful save, the view marks affected digests as stale:
`UPDATE digest_status SET stale = TRUE WHERE name IN (...)`. This is the only
write-path cost.

Read path (conditional GET for a digest):
1. Check stale flag for the requested digest (one PK lookup).
2. If not stale: normal If-Modified-Since check against file mtime, return 304
   or 200.
3. If stale: regenerate the digest, then serve the new file. Regeneration
   proceeds as follows:
   a. Generate the digest and write it to a temp file (gzip-compressed).
   b. `os.rename()` the temp file to the digest path (POSIX-atomic).
   c. Clear the stale flag: `UPDATE digest_status SET stale = FALSE WHERE
      name = ? AND stale = TRUE`. The `AND stale = TRUE` acts as a
      compare-and-swap so that concurrent readers don't regenerate twice.

   This ordering ensures that on any crash, the stale flag may over-report but
   never under-report: a digest file on disk is always complete and valid.

All digest generation logic lives in `apps/base/digests.py`, since digests
often span multiple domain tables (e.g., a prelievi write also updates
`parcel_year_production.json` used by bosco).

Digest files are gzip-compressed (stored as `.json.gz`) and served with
`Content-Encoding: gzip`.

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

When a domain page is loaded for the first time (empty cache), a modal displays
"Caricamento..." until the initial data fetch completes. Subsequent cache
refreshes happen silently in the background.

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
1. JS fetches the form HTML from Django (including the CSRF token and an
   idempotency nonce as a hidden field).
1. The form is rendered in the current page (it replaces the current view). The
   URL *does not change*. This is the one exception to the "canonical
   representation of view state" rule, since we never need to share the input
   form.
1. Client-side JS validation provides immediate feedback: the submit button is
   inactive until JS validation passes.
1. On submit, JS intercepts and forwards the POST request (including the CSRF
   token and idempotency nonce), and waits for a response.
1. The server checks the nonce: if it has already been used (i.e., a previous
   request with the same nonce succeeded), it returns the original success
   response without writing again. This prevents duplicate records when the
   network drops between server-commit and client-receive and the client retries.
   Used nonces are stored in the database with a timestamp. The nightly backup
   cron job also prunes nonces older than 24 hours.
1. The server provides authoritative validation.

The server response has one of three values. The payload is always JSON.

1. Success: Code = 200 OK, payload = { data_id: X, row_id: Y, "record": [
   Y, ... ] }

   The client updates row Y of entry X in the case with the new record and
   refreshes the tabular display.

   Note that a future background conditional GET might refresh other cache
   entries, such as those corresponding to digested data for graphs. Concrete
   example: user enters data corresponding to a new harvest operation. The
   tabular display of harvest operations updates immediately. Digested data for bar chart of
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
- A map type selector. Buttons for OSM, Topo, Satellite. Satellite is the
  default.
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

Here we describe the core relational tables that underpin the app. Per-domain
JSON digests appear in the detailed descriptions below. All tables have implicit
version (int), created_at, and modified_at columns that we omit for clarity.

- user: extends AbstractUser with (role:string, login_method:string)
  - role is one of 'admin', 'writer', 'reader'.
  - login_method is one of 'password', 'oauth'.
  - Inherits from AbstractUser: username, password, email, first_name,
    last_name, is_active, date_joined.
  - AUTH_USER_MODEL = 'apps.base.User' must be set before the first migration.

- region: (id:int, name:string)
  - Denotes a forest region or "compresa".

- eclass: (id:int, name:string, coppice:bool, min_harvest_volume:int)
  - Represents a parcel economic class. It may be coppice or high forest
    (coppice=false).
  - Characterized by a minimum volume (m3/ha) before harvesting is permitted.

- harvest_plan: (id:int, year_start:int, year_end:int, description:text)
  - Denotes a multi-year harvest plan, comprising all or most parcels.

- harvest_plan_item: (id:int, harvest_plan_id:int, parcel_id:int, year:int,
  quintals:int)
  - Denotes a calendar item in the harvest plan, i.e., that the given parcel
    will be cut in the given year, with a goal of quintals mass.

- harvest_detail: (id:int, description:text, interval:int)
  - A reusable harvest instruction, e.g., "Preferentially cut white firs of
    diameter 20-40cm". 'interval' denotes the harvest interval for coppice
    parcels (nullable for non-coppice).

- parcel_plan_detail: (harvest_plan_id:int, parcel_id:int,
  harvest_detail_id:int) — PK is (harvest_plan_id, parcel_id)
  - Maps a harvest detail to a parcel within a plan.

- parcel: (id:int, name:string, region_id:int, class_id:int, area_ha:int,
  ave_age:int, location_name:string, altitude_min_m:int, altitude_max_m:int,
  aspect:str, grade_pct:int, desc_veg:string, desc_geo:string,
  harvest_plan_id:int)
  - Represents a forest parcel. 'name' is typically an alphanumeric string like
    '11' or '2a'.
  - 'area_ha' is surface area in hectares.
  - 'ave_age' is the average age of trees in the parcel.
  - altitudes are in meters.
  - 'desc_veg' and 'desc_geo' are strings that describe the vegetative and
    geologic state of the parcel, respectively.
  - 'harvest_plan_id' (nullable) points to the parcel's current harvest plan.

- sample_area: (id:int, number:int, parcel_id:int, lat:real, lng:real,
  altitude_m:int, plan_year:int)
  - Represents a sample area.
  - 'parcel_id' maps to the parcel that this sample area was recorded as
    belonging to. Due to human errors, lat/lng may not be within the bounds of
    the stated parcel. (Both sets of data are present to allow finding these
    errors automatically in the future.)
  - 'plan_year' denotes the year of the harvest plan that the sample was used for.

- tractor: (id:int, manufacturer:string, model:string, year:int, active:bool)
  - Represents a tractor. 'year' denotes date of manufacture.
  - Retired tractors have active=false.

- crew: (id:int, name:string, notes:string, active:bool)
  - Represents a team of workers, e.g., a group of lumberjacks.

- species: (id:int, common_name:string, latin_name:string, active:bool)
  - Represents a tree species.

- preserved_tree: (id:int, species_id:int, region_id:int, parcel_id:int,
  lat:real, lng:real, note:string)
  - Represents a tree that has been marked for preservation (should not be
    harvested). Used for the "Piante ad accrescimento indefinito" view.

- optype: (id:int, name:string)
  - Implements an extensible enum: (1: "Tronchi", 2: "Cippato", 3: "Ramaglia",
    4: "Pertiche-Puntelli", 5: "Pertiche-Tronchi")

- note: (id:int, name:string)
  - Implements an extensible enum: (1: "PSR", 2: "Fitosanitario", 3:
    "Catastrofate")

- harvest_op: (id:int, date:string /* ISO 8601 */, optype_id:int, parcel_id:int,
  crew_id:int, record1:int, record2:int, quintals:float, note_id:int,
  extra_note:text)
  - Denotes a harvest operation by one crew on a given day.
  - Record1 and record2 are optional and indicate the id on a paper
    bill-of-goods form provided by the crew. They correspond to "vdp" and "prot"
    in the mannesi.csv file, respectively.

- harvest_species: (harvest_op_id:int, species_id:int, percent:int) — PK is
  (harvest_op_id, species_id)

- harvest_tractor: (harvest_op_id:int, tractor_id:int, percent:int) — PK is
  (harvest_op_id, tractor_id)

  Harvest_species and harvest_tractor denote the production breakdown of a
  single harvest operation. The percentages for species and tractors must each
  sum to 100, enforced by client-side JS validation and server-side Django
  validation (not by SQL constraints).

  Deleting a harvest_op cascades to its harvest_species and harvest_tractor
  rows. Crews, tractors, and species are never deleted — they are deactivated
  via their `active` flag.

# Internationalization

The app initially only supports Italian. The URL paths are in Italian also.

But there are no inline strings in the code. All are named constants (in both
Python on the backend and JS on the front-end) to make a future
internationalization easier. Path names are also named constants. (The
assumption is that any future language choice will be at the level of the entire
app, not individual users.)

Numbers and dates are represented using Italian locale.

Exported CSV uses semi-colons as separators.

The code itself (variable names, function names, etc.) is all in English. We
use terms like 'coppice' in the code instead of 'ceduo', and so on.

# Mobile

The app is usable on mobile in portrait mode (without needing to switch to landscape).

More on this is in the detailed description below.

# Project structure

## Directory layout

    abies/
    ├── manage.py
    ├── config/                         # Django project settings
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    ├── apps/                           # Django apps
    │   ├── base/                       # shared models, shell, common JS/CSS
    │   │   ├── models.py               # region, eclass, parcel, sample_area,
    │   │   │                           # crew, tractor, species, optype, note
    │   │   ├── views.py                # shell view, login view
    │   │   ├── urls.py
    │   │   ├── migrations/
    │   │   ├── digests.py              # all JSON digest generation
    │   │   ├── templates/base/
    │   │   │   ├── shell.html          # the long-lived SPA shell
    │   │   │   └── login.html
    │   │   └── static/base/
    │   │       ├── css/common.css
    │   │       ├── js/
    │   │       │   ├── app.js          # boot: imports domains, inits router
    │   │       │   ├── router.js       # pushState / popstate / route table
    │   │       │   ├── cache.js        # in-memory cache, conditional GETs
    │   │       │   ├── api.js          # fetch helpers
    │   │       │   ├── forms.js        # form fetch, intercept, validate, submit
    │   │       │   ├── modals.js       # error and help modals
    │   │       │   ├── table.js        # sortable-table wrapper + CSV export
    │   │       │   └── strings.js      # Italian UI string constants
    │   │       └── vendor/             # vendored: Leaflet, Chart.js,
    │   │                               # sortable-table, DM Sans
    │   ├── prelievi/                   # Prelievi domain
    │   │   ├── models.py               # harvest_op, harvest_species, harvest_tractor
    │   │   ├── views.py                # JSON endpoints, form fragments, POST
    │   │   ├── urls.py
    │   │   ├── templates/prelievi/
    │   │   │   └── _form.html          # add/edit form fragment
    │   │   └── static/prelievi/
    │   │       ├── css/prelievi.css
    │   │       └── js/prelievi.js      # PrelieviPage class
    │   ├── bosco/                      # Bosco domain (Stage 2)
    │   ├── controllo/                  # Audit domain
    │   └── impostazioni/               # Settings domain
    ├── ingest/                         # one-time ETL / data import scripts
    ├── data/                           # host-mounted runtime dir (gitignored)
    │   ├── db.sqlite3
    │   ├── digests/                    # pre-computed JSON files
    │   └── geo/                        # GeoJSON, satellite imagery
    ├── docs/
    ├── test/
    ├── Dockerfile
    ├── Makefile
    └── requirements.txt

Apps are organized under `apps/` with dotted paths in `INSTALLED_APPS` (e.g.,
`apps.prelievi`). `config/` holds the Django project settings.

## JS module conventions

Client-side code uses ES modules (`<script type="module">`). The shell page
loads `base/js/app.js` as its sole entry point. `app.js` imports shared modules
(router, cache, api, forms, etc.) and each domain's page module.

All domain modules are loaded at boot, not lazy-loaded per tab click. Total JS
is small (order of tens of KB), and paying the load cost once per session keeps
tab switching instant.

Each domain module exports a page class with a known interface: `mount(params)`,
`unmount()`, `onQueryChange(params)`. `app.js` registers these in a static route
table keyed by URL path.

Release builds use minified JS, produced by a `make minify` Makefile target.

## Static file conventions

Each Django app owns its static files under `static/<app_name>/`, following
Django's default namespacing convention. Templates follow the same pattern
(`templates/<app_name>/`). Common CSS and JS live in `base`.

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

No browser-based E2E testing framework in v1. UI is verified by manual smoke
testing.

The test instance of Abies is deployed locally and does not use Docker, to speed
up testing and debugging.

# Development environment

The dev instance runs directly on the host (no Docker) using `manage.py
runserver` and a Python virtualenv (latest Python, currently 3.14).

The `data/` directory at the project root holds `db.sqlite3`, `digests/`, and
`geo/`, identically to production but without Docker mounts.

Makefile targets for dev setup:

- `make migrate`: run Django migrations to create/update tables.
- `make import`: run ETL scripts (`ingest/`) to populate `db.sqlite3` from
  existing CSVs in `bosco/data/`.
- `make geo`: import geo data into `data/geo/` using existing bosco scripts.
- `make digest`: precompute all JSON digests. Calls `apps/base/digests.py`
  to avoid divergence with the lazy-on-read production path. Also useful for
  debugging digest generation outside of the request cycle.
- `make admin`: create the initial admin user (prompts for username/password).
- `make dev`: runs `migrate`, `import`, `geo`, `digest`, and `admin` in
  sequence — one command to go from zero to a working dev instance.

# Relationship to existing "bosco" apps

The Forest Visualization domain of Abies subsumes "boscoscopio" and certain
other "bosco" apps. The "bosco" apps remain unchanged for now but will
eventually be taken offline and replaced entirely by Abies.

# Build order

v1 ships in two stages. Stage 1 is the production MVP; Stage 2 follows on a
separate track.

## Stage 1: Prelievi MVP

This is the first production release. It exercises every major subsystem of the
architecture (auth, shell, router, cache, conditional GETs, form injection,
optimistic locking, audit, CSV export, ETL) against the simplest UI, and it is
the domain where real users can begin entering data earliest.

Stage 1 includes:

- Shell page, header, client-side router, client cache, error modal, CSV export
  button.
- Authentication (both password and MS 365 OAuth) via django-allauth +
  django-axes.
- `base` app: common models (region, eclass, parcel, crew, tractor, species,
  optype, note, harvest_op, harvest_species, harvest_tractor), ETL scripts for
  initial data import, JSON digest generation, and shared templates/static/CSS.
- `prelievi` app: the Prelievi page (sortable-table + date range slider),
  add/edit/delete form, and the prelievi.json digest.
- `controllo` app: the audit page. Nearly free once django-simple-history is
  wired up, so include it in Stage 1 so staff can see their own activity as
  they start using the app.
- `impostazioni` app: all Settings sections (Personal settings,
  Crews/Tractors/Trees, App Users). App Users is required in Stage 1 so that
  early beta users can be onboarded.
- Bosco tab: placeholder content (e.g., a link to the existing Boscoscopio app)
  so the tab is not broken.

Until Stage 2 ships, the landing page after login is Prelievi, not Bosco.

## Stage 2: Bosco

A port of Boscoscopio into the Abies shell, with the additions described in the
Bosco page section below (aree di saggio, piante ad accrescimento indefinito,
bookmarkable URLs). Novelty is low relative to Stage 1 because much of the
code can be lifted from the existing bosco/b app, so Stage 2 can proceed
independently once the Stage 1 architecture is stable.

When Stage 2 ships, the landing page reverts to Bosco.

## Future stages

The remaining domains (Segheria, Biomassa, Fotovoltaico, Rifornimenti) are
separate post-v1 stages and are not scoped here.

# Detailed description

The following sections describe the login page and then each of the v1 domain
pages, in the order in which the corresponding tabs appear in the desktop
version from left to right (or in the selection menu on mobile, top to bottom).

## Login page

The login page is mostly blank. In the center, inside a dark-green-bordered
square, is the company logo and name, entry fields for username and password,
and a "Log in with Microsoft button".

Upon successful login, users land on the Forest Visualization page.

## Bosco page

The "bosco" page is the initial landing page of Abies. In version 1 it is close
to a clone of the Boscoscopio app, with the addition of data from other Bosco
apps ("aree di saggio", "piante ad accrescimento indefinito") and bookmarkable
URLs.

- Path: /abies/bosco
- Query parameters:
  - Map type (mt={o,t,s} (OSM, Topo, Sat))
  - Map center (Lat/lng) (mc=NN.NNNNNN,NN.NNNNNN)
  - Map zoom level (mz=NN)
  - Region (c={[region name]})
  - Mode (m=N): denotes display mode (see below)
  - Parameter (q=N): denotes parameter to display (see below)
  - Parcels (p=...): denotes parcels to display (see below)
  - Species (s=...): denotes species to display (see below)
  - Dates (d1=YYYYMMDD, d2=YYYYMMDD): used for historical comparisons (see below)
  - Boolean flags (false if not present):
    - Cadastral area (fc): whether to display cadastral or computed area ("Aree
      catastali" in Boscoscopio)
    - Parcel averages (fa): whether satellite values should be rendered
      per-satellite-pixel or average per parcel ("medie per particella" in
      Boscoscopio)

### Visual appearance

The page layout is as described under "Maps" above. The lower part of the
control panel has radio buttons titled:

- Caratteristiche (default)
- Evoluzione
- Aree di saggio
- Piante ad accrescimento indefinito

When each button is selected, the rest of the control panel looks as follows:

- Caratteristiche

  Panel contains pull-down with the same list of features as those in the
  "Visualizza caratteristiche" part of Boscoscopio. The behavior is identical.
  Below the pulldown is a checkbox for "Aree catastali".

  Map shows heatmap per pixel or per parcel, depending on type of data and
  whether "media per particella" is checked, identically to Boscoscopio.
  Color range is yellow-green (low values) to dark-green (high values), except for raw normalized satellite data (0 = black -> 1 = white).

- Evoluzione

  Panel contains pull-down with the same list of features as in the "Visualizza differenze" part of Boscoscopio, plus pull-downs for two dates (years, or year-months) to compare. The behavior is identical to Boscoscopio with "limita al bosco" always set to true.

  Below the pull-downs are checkboxes for "media per particella" and "aree catastali".

  Map shows red-to-green heatmap showing (new - old) values per pixel, or
  average diffs per parcel if "media per particella" is selected. High values
  map to dark green, low values map to dark red, white in the middle.

- Aree di saggio

  Panel contains scrollable list of  parcels for the current region, identical
  to bosco/ads (but for only the current region, not all regions). There is a
  checkbox to the left of each parcel name, and the number of contained sample
  areas in parentheses to the right. Below the panel of parcels, there are
  "mostra tutte" and "nascondi tutte" buttons.

  Map displays parcel borders and yellow dots corresponding to the sample areas.

- Piante ad accrescimento indefinito

  Shows two scrollable lists, like bosco/pai. Top list is parcels in the current
  region, identical to "aree di saggio" above. Lower list is species to display.
  Each species has a checkbox (to display or not, a color-coded dot, a name, and
  a count in parentheses, identically to bosco/pai.) Both lists have "mostra
  tutte" and "nascondi tutte" buttons with the obvious semantics.

  Map displays parcel borders and colored dots corresponding to the trees.

### Query parameter details

- Caratteristiche
  - m=1
  - q=1-14 (corresponding to the entries in the pulldown menu)
  - fc=1 if "aree catastali" is checked

- Evoluzione
  - m=2
  - q=1-4 (corresponding to the entries in the pulldown menu)
  - d1=YYYYMMDD,d2=YYYYMMDD (start and end dates of comparison).
    
    If the granularity of data is year, then dates are of the form YYYY0101.
    If the granularity is month, then the dates are of them form YYYYMM01.

  - fa=1 if "media per particella" is checked.
  - fc=1 if "aree catastali" is checked.

- Aree di saggio
  - m=3
  - p=[comma-separated list of parcels (e.g., "1,2,4a,4b,14,15c")]

- Piante ad accrescimento indefinito
  - m=4
  - p=[comma-separated list of parcels (e.g., "1,2,4a,4b,14,15c")]
  - s=[comma-separated list of down-cased species names, with spaces replaced by
    underscores, e.g., "abete_rosso,castagno,betulla_bianca"]

### Data tables

Statistical data:
- parcels.json: JSON version of the parcel table (columns TBD)
- sample_areas.json: JSON version of the sample_area table
- preserved_trees.json: JSON version of the preserved_tree table
- parcel_year_production.json: a digest that conceptually is a "SELECT region,
  parcel, year, SUM(quintals) FROM harvest GROUP BY region, parcel, year", organized like the timeseries.json files in Boscoscopio.

Map data:
- particelle.geojson as in Boscoscopio
- satellite data as in Boscoscopio

## Prelievi page

The prelievi page supports recording and display of harvesting operations.

- Path: /abies/prelievi
- Query parameters:
  - Sort column: sc=N
  - Sort order: so=0/1 (ascending/descending)
  - Filter: f=(URL-encoded version of sortable table search string)

### Visual appearance

In v1, this page simply displays all harvest operations in a sortable-table,
exactly as described in "Tabular data" above.

The full dataset is served as a single compressed JSON digest. All filtering
is client-side: a double-ended date slider (see bosco/a/range-slider.*) with
year granularity restricts the displayed range, and the search box filters
within that range. No server round-trips for filtering.

Columns are:
Data, Compresa, Particella, Squadra, VDP, Q.li, Note, Altre note, (quintal columns by species in alphabetical order), (quintal columns by tractor in alphabetical order).

The add/edit form displays:

- a date picker;
- pull-downs for "Compresa", "Particella", "Squadra", "Note";
- short text input for "VDP" (verification: must be unique among cached values
  client-side, and unique across all values server-side);
- longer text input for "Altre note";
- numerical input for "Q.li" (verified to be a float).

For values by species and tractor, the form requires not quintals but
percentages.

For these fields, we list all possible choices and provide next to each one both
a numerical entry text input and a "100%" button for the common case.

Specie:
Abete: [box] [100%] Castagno: [box] [100%] ...

Trattori:
Fiat: [box] [100%] Volvo: [box] [100%] ...

Pressing a button sets the corresponding box to 100 and the others to 0.

The form has two submit buttons: "Salva" (save and return to the table view)
and "Salva e aggiungi" (save and present a blank form for the next entry).
"Salva e aggiungi" supports the common batch-entry workflow where office staff
enter a stack of paper slips in sequence.

### Data tables

- parcels.json: as above
- crews.json: JSON version of the crew
- prelievi.json: a de-normalized version of the harvest table, containing
  everything in the sortable-table as well as percentage values (to support
  prepopulating the edit form).

Successful writes on the backend cause regeneration of the
parcel_year_production.json file used by the bosco page.

## Audit page ("Controllo")

Path: /abies/controllo

This page is visible to all users.

The audit page displays a sortable-table table with the following columns:

- time and date, user, table name, action (insert/edit/delete), value before, value after

This information comes from django-simple-history. The table is not editable,
but it is searchable and sortable like all other sortable-tables.

## Settings page ("Impostazioni")

Path: /abies/impostazioni

The settings page contains several collapsible sections separated by horizontal
rules. All sections are collapsed by default. Not all sections are visible to
all users (details below): if a section is not visible, it is completely hidden,
not just collapsed.

### Personal settings

This section is visible to all users (reader, writer, admin) who use password
authentication. It provides two simple text-entry fields, "new password" and
"repeat new password". They must of course match.

### Crews, tractors, and trees

This section is visible only to writers.

They can create and edit workers, crews, tractors, and tree species.

Each of these entities is configured in its own collapsible section.

Each section contains a corresponding sortable table.

Each of these sortable tables supports adding and editing entities, but not
removing them.

In each table the rightmost column is titled "active" and denotes whether the
entity (crew, tractor, etc.) should appear as an option in new input forms.

Above each table, on the right of the search box, is a checkbox for "Only
active". It is checked by default to avoid clutter.

The tables differ in the columns that they display (and therefore the data entry
fields that the corresponding input modal provides):

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
