# La Foresta

Soc. Agr. La Foresta S.r.l. is the original Abies installation. Its users
include the office staff and field operators of a 40-person lumber company
in Calabria, Italy.

## Hosted instances

Both instances run on the same VM with separate data, static files, and
container settings.

| Instance | URL | Compose environment | Loopback port |
| --- | --- | --- | --- |
| Production | `https://abies.laforesta.it/` | `compose/.env.prod` | `127.0.0.1:8000` |
| Remote dev | `https://abies-dev.laforesta.it/` | `compose/.env.dev` | `127.0.0.1:8001` |

The deployment machine uses the `vm-abies` Docker context with
`ssh://maxp@abies.laforesta.it`. The Make targets that use SSH or rsync
default to `REMOTE=maxp@abies.laforesta.it`.

The `system` repository provisions the host with `ansible/foresta.yml`.
Its `ansible/templates/apache2-sites/abies-ssl.conf.j2` template configures
the Apache vhosts and the `/static/` cache policy for both instances.
The playbook also schedules the production backup at 23:00 host time and
mirrors it to `onedrive:abies-prod/backups` after a successful backup.

Use the instance URL from the table when enrolling Ipso devices. The current
app is served under `/ipso/` on that host. The former standalone deployment
used `https://ipso.laforesta.it/`. The [hosted deployment guide](deployment.md)
covers app deployment and device enrollment. The [backup guide](backups.md)
covers retention and restores.

## Legacy data

The [La Foresta converter](../../abies-init/README.md) prepares the canonical
CSV bundle for [bootstrap](bootstrap.md) and copies the branding assets for
install and deploy. It emits `H_measured=false` for calculated heights in the
Sabatino survey,
`H_measured=true` for measured heights in the Luca survey, and
`H_measured=false` for the current preserved-tree (PAI) data.

The historical `pdg-2026` height regressions came from samples taken for that
harvest plan. For example, its regressions CSV contained this row.

```csv
compresa,genere,funzione,a,b,r2,n
Capistrano,Abete,ln,7.0306,-4.2563,0.6312,48
```

Abies adapted the logarithmic fit from `LogarithmicRegression` in
`pdg-2026/pdg/computation.py`. The current
[height-equation import](hypsometry.md#reviewed-batch-imports) can update the
equations independently of harvest plans. The
[harvest plan CSV importer](page-piano-di-taglio.md#csv-files) accepts
the planning tool's output.

## High-resolution elevation data

These regional 1 m datasets are available through the EU data portal for the
[terrain-model work](terrain-models.md).

- [Calabria first-return digital surface model](https://data.europa.eu/data/datasets/m_amte-299fn3-1c04aa0a-5c79-466c-a266-b15428703449)
- [Calabria last-return digital surface model](https://data.europa.eu/data/datasets/m_amte-299fn3-aa01d612-8741-4340-e7ef-4b7a8d5a07ec)
- [Calabria digital terrain model](https://data.europa.eu/data/datasets/m_amte-299fn3-2cc40856-4a7e-4c84-e96b-02c2d54b674e)

The Sabatino survey is the proposed field reference for comparing
LiDAR-derived canopy volume with measured wood volume.
