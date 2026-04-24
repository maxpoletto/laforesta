# Database model

Core relational tables that underpin Abies.  Per-domain JSON digests appear
in the individual page docs under `docs/pages/`.  All tables have implicit
`version` (int), `created_at`, and `modified_at` columns that we omit below
for clarity.

Note that all fields referred to as 'optional' are nullable.

## App metadata

- user: extends AbstractUser with (role:string, login_method:string)
  - role is one of 'admin', 'writer', 'reader'.
  - login_method is one of 'password', 'oauth'.
  - Inherits from AbstractUser: username, password, email, first_name,
    last_name, is_active, date_joined.
  - AUTH_USER_MODEL = 'apps.base.User' must be set before the first migration.

## Geographic concepts

- region: (id:int, name:string)
  - Denotes a forest region or "compresa".

- eclass: (id:int, name:string, coppice:bool, min_harvest_volume:int)
  - Represents a parcel economic class. It may be coppice or high forest
    (coppice=false).
  - Characterized by a minimum volume (m3/ha) before harvesting is permitted.

- parcel: (id:int, name:string, region_id:int, eclass_id:int, area_ha:int,
  ave_age:int, location_name:string, altitude_min_m:int, altitude_max_m:int,
  aspect:str, grade_pct:int, desc_veg:string, desc_geo:string)
  - Represents a forest parcel. 'name' is typically an alphanumeric string like
    '11' or '2a'.
  - 'area_ha' is surface area in hectares.
  - 'ave_age' is the stated (official) average age of trees in the parcel. It
    may differ from the computed average of the age of sampled trees.
  - altitudes are in meters.
  - 'desc_veg' and 'desc_geo' are strings that describe the vegetative and
    geologic state of the parcel, respectively.

- sample_area: (id:int, number:int, parcel_id:int, lat:real, lng:real,
  altitude_m:int, r_m:int, group:string, note:string)
  - Represents a sample area for dendrometric measurements.
  - 'number' is a manually assigned identifier, which must be unique within a
    parcel but need not be unique across parcels.
  - The sample area is always a circle of radius r_m meters centered at (lat,
    lng).
  - Due to measurement errors (e.g., at parcel boundaries), (lat, lng) may not
    be within the bounds of the stated parcel. (Both sets of data are recorded
    to allow finding these errors automatically in the future.)
  - 'group' (optional) is a label that can be used to group a set of sample
    areas (e.g., a project name).
  - 'note' (optional) is an additional arbitrary string (e.g., for recording
    local conditions).

## Trees

- tree: (id:int, species_id:int, year:int, lat:real, lng:real, parcel_id:int,
  preserved:bool, coppice:bool)
  - Denotes a tree over time. Lat/lng may be null, or may not fall within the
    bounds of the given parcel due to measurement error (e.g., near a parcel
    border).
  - Year is (estimated) birth year.
  - If a tree is denoted as preserved, it cannot be marked for felling.
  - Coppice is true if this tree has coppice morphology. There may be coppice
    trees in a non-coppice parcel and vice-versa.

This table may be initialized in bulk via import of existing CSV files (e.g.,
alberi-calcolati.csv).

Due to error in GPS measurements and transience of tree markings (spray paint,
etc.), it is possible for the same physical tree to appear more than once, for
example because it is part of a sample one year but is then marked for cutting
some years later (see samples and marks below). We don't believe this is a
problem: samples are used to estimate per-hectare biomass averages, and later
marks measure actual (non-extrapolated) biomass to be harvested.

## Operations on specific trees

### Samples

- sample: (id:int, sample_area_id:int, date:string /* ISO 8601 */,
  harvest_plan_id:int)
  - 'harvest_plan_id' (optional) denotes the harvest plan that the sample was
    used for.

- tree_sample: (sample_id:int, tree_id:int, shoot:int, number:int, d_cm:int,
  h_m:int, L10_mm: int)
  - Denotes the findings about a particular tree during a particular sampling
    operation. PK is (sample_id, tree_id, shoot).
  - shoot is a sequential 1-based counter identifying shoots from a single coppice stump (identified by tree_id). 0 for non-coppice.
  - number is a 1-based externally assigned counter of trees within a sample.
  - L10_mm denotes the width, in mm, of the outer ten rings of the sampled tree.
  - Decoupling trees from tree samples allows us to monitor tree growth over
    time, if desired.

### Marks

- mark: (id:int, parcel_id:int, date:string /* ISO 8601 */, harvest_plan_item_id:int)
  - Represents an operation ("martellata" in Italian) during which an agronomist
    marks trees for upcoming felling. The date year and parcel should correspond
    to an existing harvest_plan_item. (If they do not, the condition is
    highlighted in the UI, but exceptions do sometimes occur, so consistency
    should not be enforced at the schema level.)
  - Note that a mark is tied to a specific harvest plan item ("cutting parcel P
    in year Y"), whereas a sample may be more generally associated with an entire harvest plan (or none at all).

- tree_mark: (mark_id:int, tree_id:int, d_cm:int, h_m:int)
  - A tree being marked for felling. Primary key (mark_id, tree_id).
  - Diameter (in cm) and height (in m) indicate size at time of marking.

## Harvests (cutting operations)

- harvest: (id:int, date:string /* ISO 8601 */, mark_id:int, lumber_id:int,
  crew_id:int, record1:int, record2:int, quintals:float, note_id:int,
  extra_note:text)
  - Denotes a cutting/harvesting operation by one crew on a given day.
  - mark_id ties the harvest back to the mandatory pre-harvest mark, which also
    identifies the parcel in question.
  - lumber_id denotes the type of produced material (logs, wood chips, etc.).
  - record1 and record2 are optional and indicate the id on a paper
    bill-of-goods form provided by the crew. They correspond to "vdp" and "prot"
    in the mannesi.csv file, respectively.

- harvest_species: (harvest_id:int, species_id:int, percent:int) — PK is
  (harvest_id, species_id)

- harvest_tractor: (harvest_id:int, tractor_id:int, percent:int) — PK is
  (harvest_id, tractor_id)

  Harvest_species and harvest_tractor denote the production breakdown of a
  single harvest operation. The percentages for species and tractors must each
  sum to 100, enforced by client-side JS validation and server-side Django
  validation (not by SQL constraints).

  Deleting a harvest cascades to its harvest_species and harvest_tractor
  rows. Crews, tractors, and species are never deleted — they are deactivated
  via their `active` flag.

## Harvest plans

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

## Other parameters

Most of these are configurable in the Settings section.

- tractor: (id:int, manufacturer:string, model:string, year:int, active:bool)
  - Represents a tractor. 'year' denotes date of manufacture.
  - Retired tractors have active=false.

- crew: (id:int, name:string, notes:string, active:bool)
  - Represents a team of workers, e.g., a group of lumberjacks.

- species: (id:int, common_name:string, latin_name:string, sort_order:int,
  active:bool)
  - Represents a tree species.
  - sort_order controls display ordering everywhere species appear. Catch-all
    entries like "Altro" use a high value (999) to sort last.

- lumber: (id:int, name:string)
  - Implements an extensible enum of harvested lumber types: (1: "Tronchi", 2:
    "Cippato", 3: "Ramaglia", 4: "Pertiche-Puntelli", 5: "Pertiche-Tronchi")

- note: (id:int, name:string)
  - Implements an extensible enum: (1: "PSR", 2: "Fitosanitario", 3:
    "Catastrofate")
