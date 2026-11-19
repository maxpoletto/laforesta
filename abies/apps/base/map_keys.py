"""Portable directory keys for initial maps built before a catalog exists."""
from uuid import NAMESPACE_URL, uuid5


def initial_map_key(region_name):
    # Part of the on-disk format. Independent builds of the same initial map
    # (including a file-only build) must agree on its directory.
    return uuid5(NAMESPACE_URL, 'urn:abies:region:' + region_name)
