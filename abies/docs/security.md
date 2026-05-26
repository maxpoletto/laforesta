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

OAuth users are pre-provisioned by an admin via the Settings page.  Each
user record carries their email; a matching verified
`allauth.EmailAddress` row is created on save.

### MS 365 OAuth configuration

The app pins OAuth to a single Entra tenant to minimise the attack
surface — only accounts within that tenant can authenticate at all.
Configuration lives in three env vars read at startup by
`config/settings.py`:

- `MS_OAUTH_CLIENT_ID` — Entra application (client) ID
- `MS_OAUTH_SECRET` — Entra application client secret
- `MS_OAUTH_TENANT` — Entra tenant ID (GUID or
  `<name>.onmicrosoft.com`).  Defaults to `common` if unset, which only
  works for multi-tenant apps.

Redirect URIs registered on the Entra app (one per deployment):
- `http://localhost:8000/accounts/microsoft/login/callback/` (local dev via `manage.py runserver`)
- `https://abies-dev.laforesta.it/accounts/microsoft/login/callback/` (shared dev VM)
- `https://abies.laforesta.it/accounts/microsoft/login/callback/` (prod)

The same Entra app handles all three; secrets can be shared (different Django
SECRET_KEYs per instance, but the OAuth client_id/secret can be reused).

In dev, export the env vars in your shell (or source a gitignored
file).  In production, supply them via a root-owned, `0600`-permission
env file read by systemd (`EnvironmentFile=`) or by Docker
(`env_file:`).

### Whitelist social adapter

`apps.base.auth.WhitelistSocialAdapter` overrides allauth's
`pre_social_login` to match incoming logins by email against the
admin-provisioned user (case-insensitive, active, login_method=oauth).
This bypasses allauth's default requirement that the incoming email be
flagged verified — Microsoft's Graph API does not assert verification,
even though email ownership is verified at account creation.  Trust is
rooted in the pinned Entra tenant.  If a new OAuth provider is added
that does not itself verify email ownership at account creation, the
adapter must be revisited.

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

The audit log is readable and searchable by all users.
