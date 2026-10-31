"""One dependency description for each mark/harvest operation family."""
from dataclasses import dataclass

from config.constants import DIGEST_FUTURE_PRODUCTION


@dataclass(frozen=True)
class DigestDependencies:
    data_ids: tuple[str, ...]

    def invalidates(self, *updated_ids):
        """Retain complete datasets updated by patches or deletes."""
        return {'data_ids': [key for key in self.data_ids if key not in updated_ids]}


def _parents(item_ids):
    return ('harvest_plan_items', DIGEST_FUTURE_PRODUCTION) if any(pk is not None for pk in item_ids) else ()


def marked_tree_dependencies(item_ids):
    ids = sorted({pk for pk in item_ids if pk is not None})
    return DigestDependencies((*[f'mark_trees_{pk}' for pk in ids], *_parents(ids), 'audit'))


def harvest_dependencies(item_ids):
    return DigestDependencies(('prelievi', 'audit', *_parents(item_ids)))
