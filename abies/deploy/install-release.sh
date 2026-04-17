#!/bin/bash
# Apply a release tarball.  Run as root on the Abies server.
#
#   sudo ./install-release.sh <tag>
#
# Expects /tmp/abies-<tag>.tar.gz (produced by `make deploy` on the
# laptop).  Override with TARBALL=/path/to/file.

set -euo pipefail

TAG="${1:?usage: install-release.sh <tag>   (e.g. 198598b)}"
TARBALL="${TARBALL:-/tmp/abies-${TAG}.tar.gz}"

[ "$(id -u)" = 0 ] || { echo "Must run as root." >&2; exit 1; }
[ -r "$TARBALL" ]  || { echo "Not readable: $TARBALL" >&2; exit 1; }

echo "==> docker load  < $TARBALL"
docker load -i "$TARBALL"

echo "==> Tagging abies:${TAG} as abies:latest"
docker tag "abies:${TAG}" abies:latest

echo "==> Restarting abies.service"
systemctl restart abies
sleep 2
systemctl --no-pager --full status abies | head -20

echo "==> Cleaning up tarball"
rm -f "$TARBALL"

echo
echo "Done.  Tail logs with:  journalctl -u abies -f"
