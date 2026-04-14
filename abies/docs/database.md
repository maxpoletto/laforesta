# Database model

Core relational tables that underpin Abies.  Per-domain JSON digests appear
in the individual page docs under `docs/pages/`.  All tables have implicit
`version` (int), `created_at`, and `modified_at` columns that we omit below
for clarity.

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

- species: (id:int, common_name:string, latin_name:string, sort_order:int,
  active:bool)
  - sort_order controls display ordering everywhere species appear. Catch-all
    entries like "Altro" use a high value (999) to sort last.
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
