# Login page

Path: /login

Centred card with company logo, username/password fields, and a
"Log in with Microsoft" button. On success, lands on the default
domain page (currently Bosco).

Microsoft callbacks require an active, admin-whitelisted OAuth user with the
same email, including when the external account was linked previously. A
rejected login returns here with a localized error and creates no authenticated
session. The username/password form remains available. The Microsoft button
appears only when the three Microsoft environment variables configure complete
credentials and a single tenant; database `SocialApp` records are ignored.
Direct login and callback requests enforce the same configuration rule even
when the button is hidden. See [security](security.md#ms-365-oauth-configuration).
