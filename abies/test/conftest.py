"""Shared test fixtures."""

import pytest
from decimal import Decimal

from apps.base.models import (
    Crew, Eclass, Note, Optype, Parcel, Region, Role, Species, Tractor, User,
)


@pytest.fixture
def regions(db):
    return [Region.objects.create(name=n) for n in ('Capistrano', 'Fabrizia', 'Serra')]


@pytest.fixture
def eclasses(db):
    items = []
    for name, coppice in [('A', False), ('B', False), ('F', True)]:
        items.append(Eclass.objects.create(name=name, coppice=coppice))
    return items


@pytest.fixture
def species(db):
    data = [
        ('Abete', 'Abies alba', 10),
        ('Castagno', 'Castanea sativa', 20),
        ('Altro', '', 999),
    ]
    return [Species.objects.create(common_name=c, latin_name=l, sort_order=o)
            for c, l, o in data]


@pytest.fixture
def tractors(db):
    return [
        Tractor.objects.create(manufacturer='Fiat', model='110-90'),
        Tractor.objects.create(manufacturer='Landini', model='135'),
    ]


@pytest.fixture
def crews(db):
    return [
        Crew.objects.create(name='Alfa'),
        Crew.objects.create(name='Beta'),
    ]


@pytest.fixture
def optypes(db):
    return [
        Optype.objects.create(name='Tronchi'),
        Optype.objects.create(name='Cippato'),
    ]


@pytest.fixture
def notes(db):
    return [
        Note.objects.create(name='PSR'),
        Note.objects.create(name='Fitosanitario'),
    ]


@pytest.fixture
def parcels(db, regions, eclasses):
    return [
        Parcel.objects.create(name='1', region=regions[0], eclass=eclasses[0],
                              area_ha=Decimal('10.5')),
        Parcel.objects.create(name='2', region=regions[0], eclass=eclasses[1],
                              area_ha=Decimal('5.0')),
        Parcel.objects.create(name='1', region=regions[1], eclass=eclasses[0],
                              area_ha=Decimal('8.0')),
    ]


@pytest.fixture
def admin_user(db):
    return User.objects.create_user(
        username='testadmin', password='testpass123!',
        role=Role.ADMIN,
    )


@pytest.fixture
def writer_user(db):
    return User.objects.create_user(
        username='testwriter', password='testpass123!',
        role=Role.WRITER,
    )


@pytest.fixture
def reader_user(db):
    return User.objects.create_user(
        username='testreader', password='testpass123!',
        role=Role.READER,
    )
