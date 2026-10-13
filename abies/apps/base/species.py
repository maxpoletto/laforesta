"""Species defaults and the new-entry/legacy selection boundary."""

from django.db.models import Q

from apps.base.models import Species, SpeciesType
from config import strings as S


OTHER_SPECIES_TYPES = {
    S.SPECIES_OTHER_CONIFER: SpeciesType.CONIFER,
    S.SPECIES_OTHER_BROADLEAF: SpeciesType.BROADLEAF,
}
PROTECTED_SPECIES_NAMES = frozenset({S.SPECIES_OTHER, *OTHER_SPECIES_TYPES})

# Only initialize older reference CSVs. Later classification is explicit and
# editable, not inferred again when an administrator changes a species name.
_CONIFER_GENERA = frozenset({
    'abies', 'picea', 'pseudotsuga', 'larix', 'pinus', 'taxus',
    'cedrus', 'cupressus', 'juniperus', 'thuja', 'sequoia', 'sequoiadendron',
    'cryptomeria', 'chamaecyparis', 'araucaria', 'podocarpus', 'tsuga',
})
_CONIFER_NAMES = frozenset(S.SPECIES_CONIFER_NAMES)
_BROADLEAF_NAMES = frozenset(S.SPECIES_BROADLEAF_NAMES)
_BROADLEAF_GENERA = frozenset({
    'acer', 'alnus', 'betula', 'carpinus', 'castanea', 'fagus', 'quercus',
    'prunus', 'juglans', 'populus', 'robinia', 'sorbus', 'tilia', 'fraxinus',
    'salix', 'ulmus', 'olea', 'ilex',
})


def initial_botanical_type(common_name, latin_name=''):
    if common_name == S.SPECIES_OTHER:
        return ''
    if common_name in OTHER_SPECIES_TYPES:
        return OTHER_SPECIES_TYPES[common_name]
    genus = (latin_name or '').strip().lower().split(' ', 1)[0]
    if genus in _CONIFER_GENERA or common_name in _CONIFER_NAMES:
        return SpeciesType.CONIFER
    if genus in _BROADLEAF_GENERA or common_name in _BROADLEAF_NAMES:
        return SpeciesType.BROADLEAF
    return ''


def selectable_species(*, existing_id=None):
    """New-entry choices, plus the original species when editing a tree."""
    choices = Q(active=True) & ~Q(common_name=S.SPECIES_OTHER)
    if existing_id is not None:
        choices |= Q(pk=existing_id)
    return Species.objects.filter(choices).order_by('sort_order', 'common_name')


def ensure_other_species():
    """Ensure all three catch-alls after a non-empty reference load.

    Defaults copy the reference species once. Later Settings edits remain
    independent; historical captured densities/measurements are untouched.
    """
    references = (
        (S.SPECIES_OTHER_CONIFER, SpeciesType.CONIFER, 'Abies alba',
         (S.SPECIES_SILVER_FIR_FULL, S.SPECIES_SILVER_FIR), '9.00', 997),
        (S.SPECIES_OTHER_BROADLEAF, SpeciesType.BROADLEAF, 'Fagus sylvatica',
         (S.SPECIES_BEECH,), '10.50', 998),
    )
    created = 0
    for name, kind, latin, names, density, order in references:
        source = (Species.objects.filter(
            Q(latin_name__iexact=latin) | Q(common_name__in=names)
        ).order_by('sort_order').first())
        _, added = Species.objects.get_or_create(common_name=name, defaults={
            'botanical_type': kind,
            'density': source.density if source else density,
            'pressler_default': source.pressler_default if source else '2.00',
            'sort_order': order,
        })
        created += added
    # Older/minimal reference files may omit Altro. Keep a stable legacy row
    # for historical imports and unclassified custom taxa, never new choices.
    _, added = Species.objects.get_or_create(common_name=S.SPECIES_OTHER, defaults={
        'sort_order': 999, 'density': '9.00', 'pressler_default': '2.00',
    })
    created += added
    return created
