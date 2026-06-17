# Security

## Permission model

Three roles: `admin`, `writer`, `reader`. Readers have read-only access.
Writers can insert/edit/delete domain data. Admins can also manage users.
Password users can change their own password.

## Authorization

MS 365 OAuth and username/password via django-allauth + django-axes
(brute-force protection). All access control is server-side.

OAuth users are pre-provisioned by an admin via the Settings page; a
matching `allauth.EmailAddress` row is created on save.

### MS 365 OAuth configuration

OAuth is pinned to a single Entra tenant (only accounts in that tenant
can authenticate). Three env vars in `config/settings.py`:

- `MS_OAUTH_CLIENT_ID` — Entra application (client) ID
- `MS_OAUTH_SECRET` — Entra client secret
- `MS_OAUTH_TENANT` — Entra tenant ID (GUID or `<name>.onmicrosoft.com`)

One Entra app registration covers all deployments (localhost, dev VM,
prod) via separate redirect URIs. The OAuth client_id/secret can be
shared; Django SECRET_KEYs differ per instance.

### Ipso upload configuration

The Abies-served Ipso PWA posts staged field uploads to the same origin.
Production deployments must set `ABIES_IPSO_UPLOAD_TOKEN`; the generated
`/ipso/upload-config.js` serves that bearer token to the installed PWA,
and `/api/ipso/uploads/` rejects requests without it.

### Whitelist social adapter

`apps.base.auth.WhitelistSocialAdapter` overrides allauth's
`pre_social_login` to match incoming logins by email against the
admin-provisioned user (case-insensitive, active, login_method=oauth).
This bypasses allauth's requirement that the email be flagged verified
— trust is rooted in the pinned Entra tenant instead.  **If a new OAuth
provider is added that does not verify email ownership at account
creation, the adapter must be revisited.**

Session expiration defaults to 12 hours (server-configurable).

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

The app records audited domain writes using django-simple-history.
High-volume tree observation rows and selected child/junction tables are
excluded from history by design; see `docs/page-controllo.md` for the
coverage contract.

The audit log is readable and searchable by all users.
