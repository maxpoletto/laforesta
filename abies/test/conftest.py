"""Shared test fixtures."""

import math
from datetime import date
from decimal import Decimal
from itertools import count

import pytest

from apps.base.models import (
    Crew, Eclass, Parcel, Product, Region, Role, Sample, SampleArea,
    SampleGrid, Species, Survey, Tractor, Tree, TreeSample, User,
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
        ('Abete', 'Abies alba', 10, Decimal('9.00'), False),
        ('Castagno', 'Castanea sativa', 20, Decimal('9.20'), False),
        ('Acero', 'Acer pseudoplatanus', 60, Decimal('9.50'), True),
        ('Altro', '', 999, Decimal('9.00'), False),
    ]
    return [Species.objects.create(common_name=c, latin_name=l, sort_order=o,
                                   density=d, minor=m)
            for c, l, o, d, m in data]


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
def products(db):
    return [
        Product.objects.create(name='Tronchi'),
        Product.objects.create(name='Cippato'),
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


@pytest.fixture
def hypso_samples(db, regions, eclasses, species):
    """A survey supporting a clean Abete fit in region 0.

    - region0 / Abete: 12 non-coppice samples on h = a·ln(D) + b.
    - region0 / Castagno: 3 samples (below a typical min_n).
    - one coppice Abete sample that compute_params must exclude.
    """
    a_true, b_true = 7.0, -4.0
    parcel = Parcel.objects.create(
        name='H1', region=regions[0], eclass=eclasses[0],
        area_ha=Decimal('5.0'),
    )
    grid = SampleGrid.objects.create(name='Hypso grid')
    area = SampleArea.objects.create(
        sample_grid=grid, parcel=parcel, number='1', lat=0.0, lon=0.0,
    )
    survey = Survey.objects.create(name='Hypso survey', sample_grid=grid)
    sample = Sample.objects.create(
        sample_area=area, survey=survey, date=date(2024, 9, 15),
    )
    counter = count(1)

    def add(sp, d_cm, coppice=False):
        tree = Tree.objects.create(species=sp, parcel=parcel, coppice=coppice)
        h = a_true * math.log(d_cm) + b_true
        TreeSample.objects.create(
            sample=sample, tree=tree, shoot=0, standard=False,
            number=next(counter), d_cm=d_cm, h_m=Decimal(str(round(h, 2))),
        )

    abete_diam = [8, 10, 12, 15, 18, 20, 24, 28, 32, 36, 40, 45]
    for d in abete_diam:
        add(species[0], d)
    for d in (10, 20, 30):
        add(species[1], d)
    add(species[0], 50, coppice=True)
    return {
        'survey': survey, 'region': regions[0], 'species': species[0],
        'a': a_true, 'b': b_true, 'n_abete': len(abete_diam),
    }
