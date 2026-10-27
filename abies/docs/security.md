# Security

## Permission model

Three roles: `admin`, `writer`, `reader`. Readers have read-only access.
Writers can insert/edit/delete domain data. Admins can also manage users.
Only admins can reopen a closed harvest-plan cantiere; normal opening and
closing remain available to writers.
Password users can change their own password.

## Authorization

MS 365 OAuth and username/password via django-allauth + django-axes
(brute-force protection). All access control is server-side.

OAuth users are pre-provisioned by an admin via the Settings page; a
matching `allauth.EmailAddress` row is created on save.

### MS 365 OAuth configuration

OAuth is pinned to a single Entra tenant (only accounts in that tenant
can authenticate). Do not use the Microsoft `/common`, `/organizations`,
or `/consumers` endpoints for Abies. Configure a tenant-specific value
(GUID or `<name>.onmicrosoft.com`) as `MS_OAUTH_TENANT`.

App credentials are configured exclusively by environment variables:

- `MS_OAUTH_CLIENT_ID` — Entra application (client) ID
- `MS_OAUTH_SECRET` — Entra client secret
- `MS_OAUTH_TENANT` — Entra tenant ID (GUID or `<name>.onmicrosoft.com`)

One Entra app registration covers all deployments (localhost, dev VM,
prod) via separate redirect URIs. The OAuth client_id/secret can be
shared; Django SECRET_KEYs differ per instance. Set all three Microsoft env vars
to enable OAuth, or omit the credentials for password-only deployments.

`apps.base.oauth.microsoft_oauth_app` is the shared configuration source for the
login button, allauth app discovery, and both the login and callback routes.
Authentication never reads database `SocialApp` records. Missing credentials or
a missing/broad tenant hide the button and reject direct authentication requests
locally with the standard localized login-denied message, before any token
exchange. A Django security check (`base.E001`) also reports invalid configured
apps; environment validation rejects incomplete credentials or a broad tenant
when loading settings. An unsupported `APPS` configuration is rejected too.

Allauth still uses an unsaved `SocialApp` Python object to represent the
environment configuration. Its empty `socialaccount_socialapp` and
`socialaccount_socialapp_sites` tables remain for compatibility with upstream
migrations and the `SocialToken.app` foreign key. They are not a supported
configuration mechanism. No table-removal migration is needed. User identity
links (`SocialAccount`) and token cleanup keep their normal allauth schema.

### Ipso-Abies integration

The Abies-served Ipso PWA uses a shared bearer secret for unauthenticated
device reference downloads and staged uploads. Production deployments must
set `ABIES_IPSO_SECRET`. See `docs/ipso-abies.md` for the data flows,
trust model, rate limiting, and deployment notes.

### Whitelist social adapter

`apps.base.auth.WhitelistSocialAdapter` overrides allauth's
[`pre_social_login`](https://docs.allauth.org/en/latest/socialaccount/adapter.html#allauth.socialaccount.adapter.DefaultSocialAccountAdapter.pre_social_login)
to check every incoming login against the current admin-provisioned whitelist
(case-insensitive email, active, login_method=oauth). This includes logins
resolved by an existing provider UID or by allauth's email lookup. The email
comes from the provider's incoming `email_addresses`, not `sociallogin.user.email`:
allauth replaces the latter with the local user's data during account lookup.
Microsoft supplies its `mail`, falling back to `userPrincipalName`.
Missing/mismatching email, an inactive user, or password-only login blocks the
OAuth callback and returns to the Abies login page with a localized message.
A linked identity cannot be reassigned to a different whitelisted user by
changing the email reported by the provider. First logins may create a link
only to a matching whitelisted user; self-registration remains closed.

This bypasses allauth's requirement that the email be flagged verified
— trust is rooted in the pinned Entra tenant instead.  **If a new OAuth
provider is added that does not verify email ownership at account
creation, the adapter must be revisited.**

Saving a changed login method or email in Settings removes that user's existing
`SocialAccount` links and their stored tokens, in the same transaction as the
User/EmailAddress changes and audit invalidation. Password-only accounts also
have any legacy links removed on save. OAuth users can link again on the next
login with their current whitelisted email. A case-only email edit preserves
links, as do ordinary name/role edits. Deactivation preserves links but blocks
login until the account is active again.

The callback check protects accounts with stale links already in the database;
no data migration is required. Link removal governs future OAuth logins.
Existing sessions follow Django's active-user and authentication-hash checks:
deactivation blocks subsequent authenticated requests, and changing a password
invalidates sessions carrying the old hash. OAuth account saves already set a
new unusable password, which also changes that hash. Removing a social link
alone does not revoke an otherwise valid session.

Session expiration defaults to 12 hours (server-configurable).

## Password policy

Django's default `AUTH_PASSWORD_VALIDATORS` are enabled: minimum length (8
characters), common password check, numeric-only check, similarity to username.

## Rate limiting

django-axes handles login brute-force protection. Authenticated API requests use
separate per-user rate-limit buckets: safe reads default to 300 requests per
minute, while mutations retain the stricter 60 requests per minute limit. This
allows normal detail-page navigation without weakening protection for data
entry. Configure the quotas with `ABIES_API_READ_RATE_LIMIT` and
`ABIES_API_WRITE_RATE_LIMIT`, and their shared window with
`ABIES_API_RATE_WINDOW_S`. Tracking is in memory and resets on process restart.

## Content security

Observation photo uploads use shared streaming byte/count limits and decoded
pixel bounds on both the desktop and Ipso endpoints. Supported raster inputs
must fully decode and permanent images are re-encoded by the server. See
[Osservazioni](page-osservazioni.md#image-policy-and-upload-limits) for defaults,
metadata handling and validation responses. Client conversion is a bandwidth
optimization; it does not establish the server's safety bounds.

The shell page sets a Content-Security-Policy header that blocks inline scripts.
All user-provided content (notes, descriptions, etc.) rendered into the DOM must
use `textContent`, never `innerHTML`, to prevent XSS.

## Auditing

The app records audited domain writes using django-simple-history.
High-volume tree observation rows and selected child/junction tables are
excluded from history by design; see `docs/page-controllo.md` for the
coverage contract.

The audit log is readable and searchable by all users.
