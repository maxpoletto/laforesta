"""Species identity and calculation policy shared by equation consumers."""
from decimal import Decimal

from django.db import connection
from django.db.models import F

from apps.base.models import Species, VolumeEquation, tree_mass_q
from apps.base.tabacchi import TABACCHI_B, TABACCHI_PROXIES, _volume_from_b


def lock_species():
    """Acquire the common equation-edit boundary inside transaction.atomic()."""
    if not connection.in_atomic_block:
        raise RuntimeError('Equation writes require transaction.atomic()')
    if not connection.features.has_select_for_update:
        Species.objects.all().update(version=F('version'))
    return list(Species.objects.order_by('pk').select_for_update(
        no_key=connection.features.has_select_for_no_key_update,
    ))


def seed_volume_equation(species):
    """Initialize a recognized new reference row once; never reset settings."""
    name = species.common_name
    # Latin identity also recognizes a renamed canonical silver fir or beech.
    latin_names = {'abies alba': 'Abete', 'fagus sylvatica': 'Faggio'}
    reference_name = TABACCHI_PROXIES.get(name, latin_names.get(
        species.latin_name.strip().lower(), name,
    ))
    coefficients = TABACCHI_B.get(reference_name)
    if coefficients is None:
        return
    proxy = None
    if name in TABACCHI_PROXIES:
        proxy = Species.objects.filter(common_name=reference_name).first()
        if proxy is None:
            latin = next((key for key, value in latin_names.items() if value == reference_name), '')
            proxy = Species.objects.filter(latin_name__iexact=latin).first() if latin else None
    values = {'source': 'Tabacchi', 'proxy_species': proxy}
    if proxy is None:
        values.update(b0=Decimal(str(coefficients[0])), b1=Decimal(str(coefficients[1])),
                      b2=Decimal(str(coefficients[2])) if len(coefficients) == 3 else None)
    VolumeEquation.objects.get_or_create(species=species, defaults=values)


def coefficients_for_species(species_id, equations=None):
    """Follow explicit ID links; never fall back to a display-name lookup."""
    seen = set()
    while species_id not in seen:
        seen.add(species_id)
        equation = (equations.get(species_id) if equations is not None else
                    VolumeEquation.objects.filter(species_id=species_id).first())
        if equation is None:
            return None
        if equation.proxy_species_id is not None:
            species_id = equation.proxy_species_id
            continue
        if equation.b0 is None or equation.b1 is None:
            return None
        return tuple(value for value in (equation.b0, equation.b1, equation.b2)
                     if value is not None)
    return None


def estimate_volume_and_mass(d_cm, h_m, species):
    if d_cm is None or h_m is None:
        return None, None
    coefficients = coefficients_for_species(species.pk)
    if coefficients is None:
        return None, None
    volume = _volume_from_b(float(d_cm), float(h_m), [float(value) for value in coefficients])
    return volume, tree_mass_q(volume, Decimal(species.density))
