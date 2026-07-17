"""Tests for landing-page validation and resolution."""

import pytest

from apps.base.landing import clean_landing_page, user_landing_page
from apps.base.models import SiteSettings


@pytest.mark.parametrize('raw, expected', [
    ('/prelievi', '/prelievi'),
    (' /bosco?mode=evoluzione#map ', '/bosco?mode=evoluzione#map'),
    ('/campionamenti/griglie?grid=1', '/campionamenti/griglie?grid=1'),
    ('/rilevamenti/griglie?grid=1', '/rilevamenti/griglie?grid=1'),
])
def test_clean_landing_page_accepts_app_routes(raw, expected):
    assert clean_landing_page(raw) == expected


@pytest.mark.parametrize('raw', [
    'prelievi',
    'https://example.com/prelievi',
    '//example.com/prelievi',
    '/api/prelievi/data/',
    '/admin/',
    '/ipso/',
    '/prelievi\\bad',
])
def test_clean_landing_page_rejects_non_app_routes(raw):
    with pytest.raises(ValueError):
        clean_landing_page(raw)


@pytest.mark.django_db
def test_user_landing_page_prefers_user_value(writer_user):
    settings = SiteSettings.load()
    settings.default_landing_page = '/bosco'
    settings.save()
    writer_user.landing_page = '/campionamenti?grid=1'

    assert user_landing_page(writer_user) == '/campionamenti?grid=1'


@pytest.mark.django_db
def test_user_landing_page_uses_site_default(writer_user):
    settings = SiteSettings.load()
    settings.default_landing_page = '/bosco'
    settings.save()

    assert user_landing_page(writer_user) == '/bosco'


@pytest.mark.django_db
def test_user_landing_page_falls_back_to_prelievi(writer_user):
    assert user_landing_page(writer_user) == '/prelievi'
