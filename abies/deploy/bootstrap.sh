#!/bin/bash
# First-time bootstrap of an Abies server.  Run as root.  Idempotent.
#
#   sudo ./bootstrap.sh
#
# Override any of the defaults by exporting before invocation, e.g.:
#   sudo ABIES_UID=20000 ABIES_DATA=/srv/abies/data ./bootstrap.sh

set -euo pipefail

ABIES_UID="${ABIES_UID:-10001}"
ABIES_GID="${ABIES_GID:-10001}"
ABIES_DATA="${ABIES_DATA:-/var/lib/abies/data}"
ABIES_STATIC="${ABIES_STATIC:-/var/lib/abies/static}"
ABIES_ETC="${ABIES_ETC:-/etc/abies}"

[ "$(id -u)" = 0 ] || { echo "Must run as root." >&2; exit 1; }

DEPLOY_DIR=$(cd "$(dirname "$0")" && pwd)

echo "==> Installing Docker if missing"
if ! command -v docker >/dev/null; then
    apt-get update
    apt-get install -y docker.io
    systemctl enable --now docker
fi

echo "==> Creating host directories"
install -d -m 0755 -o "$ABIES_UID" -g "$ABIES_GID" "$ABIES_DATA"
install -d -m 0755 -o "$ABIES_UID" -g "$ABIES_GID" "$ABIES_STATIC"
install -d -m 0755 "$ABIES_ETC"

echo "==> Installing env file template (if absent)"
if [ ! -f "$ABIES_ETC/env" ]; then
    cp "$DEPLOY_DIR/env.example" "$ABIES_ETC/env"
    chmod 0600 "$ABIES_ETC/env"
    echo "    Wrote $ABIES_ETC/env — EDIT BEFORE STARTING ABIES"
else
    echo "    $ABIES_ETC/env already exists, leaving as is"
fi

echo "==> Installing systemd unit"
cp "$DEPLOY_DIR/abies.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable abies

cat <<EOF

Bootstrap complete.  Next:

  1. Fill in $ABIES_ETC/env          (secrets + OAuth creds)
  2. Seed data:                       scp db.sqlite3 and geo/ into $ABIES_DATA
                                      then: chown -R $ABIES_UID:$ABIES_GID $ABIES_DATA
  3. Ship first release from laptop:  make deploy
     then on this server:             sudo $DEPLOY_DIR/install-release.sh <tag>
  4. Wire Apache:                     see $DEPLOY_DIR/apache-abies.conf
EOF
