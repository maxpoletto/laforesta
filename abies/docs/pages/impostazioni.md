# Settings page ("Impostazioni")

Path: /abies/impostazioni

The settings page contains several collapsible sections separated by horizontal
rules. All sections are collapsed by default. Not all sections are visible to
all users (details below): if a section is not visible, it is completely hidden,
not just collapsed.

## Personal settings

This section is visible to all users (reader, writer, admin) who use password
authentication. It provides two simple text-entry fields, "new password" and
"repeat new password". They must of course match.

## Crews, tractors, and trees

This section is visible only to writers.

They can create and edit crews, tractors, and tree species.

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

## App users

This section is visible only to admins.

Admins can create new app users and edit existing users.

The sortable-table contains the following columns:

- First and last name.
- Username.
- Email.
- Login method (one of password or OAuth).
- Created-at time.
- Active status.

Users are editable ("pencil" icon next to each row) and creatable ("plus" icon
at bottom of table).

The user input/edit form has the following fields:

- Login method radio button (password or OAuth).
- Email (required for both login methods; for OAuth, must match the
  Entra account email used for login).
- Username (required for password login only; auto-populated from email
  for OAuth users).  Hidden from the form when OAuth is selected.
- Password (repeated text input, values must match).  Hidden when OAuth
  is selected.
- First name, Last name.
- Role (pull-down menu: Membro/Redattore/Amministratore — reader/
  writer/admin).
- Active status (checkbox). Only active users can log in.

Changes take place when the admin presses the "Submit" button.

The initial admin account is configured at server installation time.

The admin must add an OAuth user (with matching email) to whitelist
them for OAuth access.

The Impostazioni tab itself is hidden for users who would see no
sections (reader + OAuth).
