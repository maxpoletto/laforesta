# UI architecture

The app is structured as a SPA-lite. After authentication, Django renders a
single shell page that persists for the duration of the session. All subsequent
navigation happens client-side without full page reloads.

## Shell and header

The shell (`apps/base/templates/base/shell_it.html`) contains:
- The header, shared by all domain pages.
- A content area (`<main id="content">`) where domain-specific content is rendered.
- A modal container (`<div id="modal-container">`).
- HTML `<template>` elements for each domain page's scaffold and modals
  (via `{% include %}` of per-app `_shell_templates_it.html` files).

The shell is rendered once and never reloads during normal use.
`<template>` content is inert (not rendered, no CSS, no scripts) until
JS clones it via `cloneTemplate(id)` (`base/js/templates.js`).

The header is adaptive for desktop and mobile. On narrow displays it contains only:
- The logo of the company.
- The name of the currently active domain.
- A hamburger icon for a menu that allows switching to other domains.

On wider displays it contains:
- The logo and name of the company.
- The names of the domains as tabs, with the currently active domain highlighted.

Tab order, left to right:

  Bosco · Piani di taglio · Campionamenti · Prelievi · Squadre · Importazione · Controllo · Impostazioni

The header is fixed in the viewport. Content scrolls beneath it.

## Routing and URLs

URLs are human readable and are the canonical representation of the view state.
They encode the domain (in the path), and any data filters, chart selection,
etc. as query parameters.

Changing domain or page-specific parameters changes the URL via
`history.pushState()` and renders the appropriate content. The back button works
via `popstate`.

All URLs are bookmarkable and shareable.

The router also remembers the last-visited URL for each domain within the
session.  Clicking a *different* tab restores that tab's last URL (preserving
its filters / sort / chart state), so jumping Prelievi → Controllo → Prelievi
returns you to where you were.  Clicking the tab you are already on goes to
its bare URL — a deliberate "reset" affordance.  Browser back / forward use
the real history stack and are unaffected.  The stash is in-memory only;
a full page reload clears it.

The paths and query parameters for each domain are documented in the
per-page specs (`docs/page-*.md`).
