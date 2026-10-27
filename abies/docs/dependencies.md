# Dependencies and updates

Abies uses reviewed dependency locks and automated advisory checks. Maintainers
respond to security alerts as they arrive and review routine update PRs monthly.
There is no manual weekly inspection requirement. Updates are tested and reviewed
before merging; deployment remains a separate operator action.

## Files and sources of truth

| File | Purpose |
| --- | --- |
| `requirements.in` | Approved production packages and version ranges |
| `requirements.txt` | Generated production lock: exact direct/transitive versions and hashes |
| `requirements-dev.in` | Test/audit tools, constrained to the production lock |
| `requirements-dev.txt` | Generated complete development/test environment |
| `vendor/package.json`, `vendor/package-lock.json` | Exact upstream sources for browser libraries |
| `Dockerfile` | Python patch release and immutable official image digest |
| `../.github/dependabot.yml` | Automated update PRs for Python, npm, Docker and Actions |
| `../.github/workflows/abies-dependencies.yml` | Tests, lock/asset checks, audits and image scans |

Compile Python locks on **Linux with Python 3.14**, matching production and CI.
The pinned `uv` tool compiles standard pip requirements files; installation
continues to use `pip install --require-hashes`.
The `.in` files are maintained by hand; the `.txt` locks are generated. Both
installation paths enforce hashes. Production installs only `requirements.txt`.
Production also requires wheels, avoiding unpinned source-build dependencies.
The complete development/test environment installs from `requirements-dev.txt`.

The container records its actual Python packages and apt packages under
`/usr/local/share/abies/`. CI also saves these inventories, the built image ID,
and the image vulnerability report as artifacts for 14 days. Apt packages obtain
current Debian security updates when the image is built, so the inventory/image
ID identifies the actual build; rebuilding a source revision can still change
those OS packages. These artifacts describe the CI build, not the deployed VM.

The build validates dependencies with pip, records the runtime inventory, then
removes pip and ensurepip (including their bundled libraries) from the final
image. The initial image scan found advisories in pip's vendored libraries even
though the app package audit was clean. Installers are build tools; changes to
production packages require a rebuilt image. `bin/check-runtime` verifies that
the final inventory matches installed packages and that native libraries import.
Development/CI installation tooling is kept current separately; the Python lock
audit does not inspect libraries bundled inside tools such as pip.
At this refresh, latest stable pip 26.2.2 still vendors the older libraries
reported by the image scan. They are absent from the runtime image; monitor
pip's upstream vendoring refresh for the development/build environment. The
scan findings alone do not establish exploitability of each vendored subset.

The initial tested refresh uses Python 3.14.10, Django 6.0.9, allauth 65.21.0,
Gunicorn 26.2.0, Pillow 13.0.0, pillow-heif 1.8.1, numpy 2.5.4, rasterio 1.5.2
and requests 2.34.2. Browser sources are Leaflet 1.9.4, Chart.js 4.6.1 and
sortable-table 1.5.0. Consult the locks for the complete package inventory.

## Initial installation

Create a Python 3.14 virtual environment and install the locked tooling and app:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-dev.txt
npm ci --ignore-scripts --prefix vendor
make check-vendor PYTHON=.venv/bin/python
```

Node 24 is used in CI. npm packages are asset sources only; browsers use the
committed files under `apps/base/static/vendor`. Roboto fonts remain managed
manually. No npm lifecycle scripts are needed.

## Updating Python dependencies

Within the approved ranges:

```sh
make upgrade-dependencies PYTHON=.venv/bin/python
.venv/bin/python -m pip install --require-hashes -r requirements-dev.txt
.venv/bin/python -m pip check
make audit-dependencies PYTHON=.venv/bin/python
make test PYTHON=.venv/bin/python
```

To reconcile inputs or regenerate locks without requesting newer versions:

```sh
make lock-dependencies PYTHON=.venv/bin/python
```

The upgrade target updates the production lock first and then constrains the
test lock to it. Review both diffs, release notes and audit results. Use a fresh
environment or remove packages no longer present when verifying an upgrade;
`pip install` alone does not uninstall obsolete packages.

Feature/major upgrades are deliberate changes. Django remains on 6.0 security
patches; its support ends in April 2027, so plan the 6.2 LTS upgrade before then.
Changing an approved range requires a compatibility/support review. Pinned
resolver/audit tools are updated through the same reviewed process.

The initial refresh deliberately moves Pillow to 13.0.0: the audit of 12.3.0
reported image-decoder advisories fixed in 13. The image/thumbnail tests cover
the APIs used by Abies; see the [Pillow release notes](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html).

## Updating browser dependencies

Edit `vendor/package.json` or review a Dependabot PR, regenerate the npm lock
when necessary, then refresh the files actually served:

```sh
npm install --package-lock-only --ignore-scripts --prefix vendor
make update-vendor
make check-vendor
npm audit --prefix vendor --omit=dev --audit-level=high
make test-js
```

Commit the manifest, lock and changed browser files together. Dependabot updates
the manifest/lock; it does not copy files into Django static directories. CI
rejects stale, missing or obsolete browser assets until they match the installed
locked packages. A check never rewrites committed files.

## Automated checks and response

Dependabot security alerts are triggered by advisories and dependency graph
changes. Enable security update PRs separately from version updates. Routine PRs
are grouped monthly; security updates do not wait for the monthly schedule.
Major library upgrades and Django feature upgrades receive deliberate review.
The Docker rule stays on Python 3.14 while allowing patch/digest updates. Action
revisions are pinned to commit SHAs and updated by Dependabot.

The workflow runs on Abies PRs/pushes, on manual dispatch, and daily at 06:23 UTC.
It verifies lock regeneration, browser assets, all Python/JS tests, migration
consistency, Python advisories and npm advisories. It builds the production
container, checks its environment/Gunicorn configuration and scans OS/library
packages with a pinned Trivy version. Image scans fail on **fixable high or
critical** vulnerabilities; unfixable and lower-severity image findings are
outside that gate. Python audit findings fail the package audit; npm's gate is
high/critical. Scan/service failures must be investigated rather than dismissed
as a clean audit. Scanners report known advisories, not a proof of safety.

If a Python PR leaves inputs, the two locks or generator formatting inconsistent,
reconcile the `.in` files and run `make lock-dependencies` before reviewing the
resulting diff. CI requires the committed files to match that regeneration.

Prioritize applicable exposed/security fixes promptly. If an alert cannot be
fixed immediately, record applicability, mitigation and its next review date.
Review routine grouped PRs monthly, including upstream support lifetimes. Test
login/OAuth, Ipso capture/upload, image handling, geospatial calculations and
backup/restore before deploying. Keep the existing single Gunicorn worker unless
its independent concurrency/rate-limit requirements are addressed.

## GitHub setup

1. Push/merge the commits into the repository's **default branch**. The root
   `.github` directory is above `abies/`; do not move it inside the app.
2. Under repository **Settings → Advanced Security**, enable **Dependency
   graph**, **Dependabot alerts** and **Dependabot security updates**. Version
   update PRs are configured by the committed `dependabot.yml`.
3. Enable GitHub Actions if disabled. If allowed actions are restricted, allow
   `actions/*`, `aquasecurity/trivy-action` and its `aquasecurity/setup-trivy`
   helper. The workflow needs read-only repository access and no secrets,
   production credentials or deployment access.
4. Under **Watch → Custom**, subscribe to **Security alerts**. Choose
   email/GitHub delivery or a digest under your account's
   [notification settings](https://github.com/settings/notifications). Enable
   Actions failure notifications for scheduled image/package checks too.
5. Run **Actions → Abies dependency checks → Run workflow** once and inspect
   both jobs and their artifacts. Review existing security alerts directly in
   the Security tab; notification preferences alone do not replace that first
   inspection.

The workflow uses path filters because this repository contains other projects.
Do not make its jobs globally required for every PR without adjusting those
filters: GitHub can leave a skipped required workflow pending on unrelated PRs.
Automatic merging/deployment is not configured.

Sources: [Dependabot detection](https://docs.github.com/en/code-security/reference/supply-chain-security/troubleshoot-dependabot/vulnerability-detection),
[notifications](https://docs.github.com/en/subscriptions-and-notifications/how-tos/managing-security-notifications),
[configuration options](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference),
[uv lock generation](https://docs.astral.sh/uv/pip/compile/),
[Trivy action](https://github.com/aquasecurity/trivy-action).
