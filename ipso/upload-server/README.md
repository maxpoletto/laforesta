# ipso-upload

Tiny stdlib HTTP server that accepts session CSVs from the ipso PWA
and stores them on disk keyed by session UUID.

## Layout on the VM

| Path                                 | Owner                | Notes                                |
|--------------------------------------|----------------------|--------------------------------------|
| `/opt/ipso-upload/server.py`         | `root:root`, `0755`  | The server binary (copy of this file)|
| `/etc/ipso-upload/config.json`       | `root:root`, `0600`  | Token + paths + rate limit           |
| `/etc/systemd/system/ipso-upload.service` | `root:root`     | systemd unit                         |
| `/var/lib/ipso-upload/uploads/`      | `ipso-upload:ipso-upload`, `0750` | Where CSVs and meta sidecars land |
| Apache vhost (ipso.laforesta.it)     | -                    | Adds `ProxyPass /upload`             |

## Local dev

    cd ipso/upload-server
    cp config.example.json config.json   # then set token + upload_dir
    make run

## Tests

    cd ipso/upload-server
    make test

## Rotating the token

Two-step rotation:

1. Update `/etc/ipso-upload/config.json` on the VM and
   `sudo systemctl restart ipso-upload`.
2. On the development host, update `ipso/secrets/upload_config.json`
   with the new token, then `cd ipso && make deploy`.

In-flight uploads that span the rotation window will return `401`;
the client treats this as a hard error and the operator can bail to
local-only delivery for that one session.

## Logs

    sudo journalctl -u ipso-upload -f

Each upload emits a single structured line including the session id,
operator, compresa, tree count, byte size, source IP, and elapsed ms.

## Retrieval

The retrieving host runs (e.g., from cron):

    rsync -av --remove-source-files \
      ipso.laforesta.it:/var/lib/ipso-upload/uploads/*.csv \
      ipso.laforesta.it:/var/lib/ipso-upload/uploads/*.meta.json \
      ./incoming/

`--remove-source-files` keeps the server directory bounded. The meta
sidecar file moves with its CSV.
