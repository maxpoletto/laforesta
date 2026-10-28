"""OAuth policy must hold after allauth resolves an existing provider identity."""

import json
from urllib.parse import parse_qs, urlparse

import pytest
import requests
from allauth.account.models import EmailAddress
from allauth.socialaccount.models import SocialAccount, SocialToken
from django.contrib.auth import SESSION_KEY
from django.test import Client
from django.urls import reverse

from apps.base.models import LoginMethod, UsedNonce, User


@pytest.fixture
def oauth_user(db):
    user = User.objects.create_user(
        username='person@example.com', email='person@example.com',
        login_method=LoginMethod.OAUTH,
    )
    EmailAddress.objects.create(user=user, email=user.email, primary=True, verified=True)
    return user


@pytest.fixture
def microsoft_login(settings, monkeypatch):
    """Run the real login/callback views, mocking only Microsoft's HTTP replies."""
    settings.SOCIALACCOUNT_PROVIDERS = {'microsoft': {
        'APP': {'client_id': 'test-client', 'secret': 'test-secret'},
        'TENANT': 'test.onmicrosoft.com',
    }}
    profile = {}

    def microsoft_response(_session, method, url, **kwargs):
        if url == 'https://login.microsoftonline.com/test.onmicrosoft.com/oauth2/v2.0/token':
            assert method == 'POST'
            body = {'access_token': 'test-token', 'token_type': 'Bearer'}
        else:
            assert url == 'https://graph.microsoft.com/v1.0/me'
            assert method.upper() == 'GET'
            body = profile
        response = requests.Response()
        response.status_code = 200
        response.headers['content-type'] = 'application/json'
        response._content = json.dumps(body).encode()
        return response

    monkeypatch.setattr(requests.Session, 'request', microsoft_response)

    def login(email, *, uid='microsoft-person', use_upn=False):
        profile.clear()
        profile.update(id=uid, mail=None if use_upn else email, userPrincipalName=email)
        client = Client()
        start = client.post(reverse('microsoft_login'))
        assert start.status_code == 302, start.content[:200]
        state = parse_qs(urlparse(start.url).query)['state'][0]
        response = client.get(reverse('microsoft_callback'), {'state': state, 'code': 'test-code'})
        return client, response

    return login


def _link(user, uid='microsoft-person'):
    return SocialAccount.objects.create(user=user, provider='microsoft', uid=uid)


def _assert_denied(client, response, user):
    assert SESSION_KEY not in client.session
    user.refresh_from_db()
    assert user.last_login is None
    assert response.status_code == 302
    assert response.url == reverse('login')
    page = client.get(response.url)
    assert page.status_code == 200
    assert 'Accesso Microsoft non consentito' in page.content.decode()


@pytest.mark.django_db
@pytest.mark.parametrize('linked', [False, True])
@pytest.mark.parametrize('use_upn', [False, True])
def test_callback_accepts_current_active_oauth_whitelist(
        microsoft_login, oauth_user, linked, use_upn):
    if linked:
        _link(oauth_user)
    client, response = microsoft_login('PERSON@example.COM', use_upn=use_upn)

    assert response.status_code == 302
    assert response.url == '/prelievi'
    assert client.session[SESSION_KEY] == str(oauth_user.pk)
    assert SocialAccount.objects.get(uid='microsoft-person').user_id == oauth_user.pk


@pytest.mark.django_db
@pytest.mark.parametrize('change', ['password', 'local-email', 'provider-email', 'no-email', 'inactive'])
def test_existing_link_does_not_bypass_current_policy(microsoft_login, oauth_user, change):
    _link(oauth_user)
    incoming_email = oauth_user.email
    if change == 'password':
        oauth_user.login_method = LoginMethod.PASSWORD
        oauth_user.set_password('secure-local-password99!')
    elif change == 'local-email':
        oauth_user.email = 'new@example.com'
    elif change == 'provider-email':
        incoming_email = 'not-whitelisted@example.com'
    elif change == 'no-email':
        incoming_email = None
    else:
        oauth_user.is_active = False
    oauth_user.save()
    password = oauth_user.password

    client, response = microsoft_login(incoming_email)

    _assert_denied(client, response, oauth_user)
    assert oauth_user.password == password
    assert SocialAccount.objects.get(uid='microsoft-person').user_id == oauth_user.pk


@pytest.mark.django_db
def test_existing_link_cannot_be_remapped_by_another_whitelisted_email(microsoft_login, oauth_user):
    _link(oauth_user)
    other = User.objects.create_user(
        username='other@example.com', email='other@example.com', login_method=LoginMethod.OAUTH,
    )

    client, response = microsoft_login(other.email)

    _assert_denied(client, response, oauth_user)
    assert SocialAccount.objects.get(uid='microsoft-person').user_id == oauth_user.pk
    assert not SocialAccount.objects.filter(user=other).exists()


@pytest.mark.django_db
def test_allauth_email_lookup_cannot_bypass_password_policy(microsoft_login, oauth_user, settings):
    # Exercise allauth's other existing-user path, without a SocialAccount yet.
    settings.SOCIALACCOUNT_PROVIDERS['microsoft']['VERIFIED_EMAIL'] = True
    oauth_user.login_method = LoginMethod.PASSWORD
    oauth_user.set_password('secure-local-password99!')
    oauth_user.save()
    password = oauth_user.password

    client, response = microsoft_login(oauth_user.email)

    _assert_denied(client, response, oauth_user)
    assert oauth_user.password == password
    assert not SocialAccount.objects.filter(user=oauth_user).exists()


def _save_user(admin_user, user, **changes):
    client = Client(raise_request_exception=False)
    client.force_login(admin_user)
    body = {
        'row_id': user.pk, 'version': user.version,
        'username': user.username, 'email': user.email,
        'login_method': user.login_method, 'role': user.role, 'is_active': user.is_active,
        'password1': '', 'password2': '', 'nonce': 'change-oauth-policy',
        **changes,
    }
    return client.post('/api/impostazioni/users/save/', json.dumps(body),
                       content_type='application/json')


@pytest.mark.django_db
@pytest.mark.parametrize('change', [
    'password', 'oauth', 'legacy-password', 'email', 'unchanged', 'case-only', 'inactive',
])
def test_account_edits_clear_only_superseded_links(admin_user, oauth_user, change):
    if change in ('oauth', 'legacy-password'):
        oauth_user.login_method = LoginMethod.PASSWORD
        oauth_user.set_password('secure-local-password99!')
        oauth_user.save()
    account = _link(oauth_user)
    token = SocialToken.objects.create(account=account, token='old-token')
    other_account = _link(admin_user, uid='other-person')
    changes = {
        'password': {'login_method': LoginMethod.PASSWORD,
                     'password1': 'secure-local-password99!', 'password2': 'secure-local-password99!'},
        'oauth': {'login_method': LoginMethod.OAUTH},
        'legacy-password': {},
        'email': {'email': 'new@example.com'},
        'unchanged': {'first_name': 'Changed'},
        'case-only': {'email': 'PERSON@example.COM'},
        'inactive': {'is_active': False},
    }[change]

    response = _save_user(admin_user, oauth_user, **changes)

    assert response.status_code == 200, response.content
    keep_link = change in ('unchanged', 'case-only', 'inactive')
    assert SocialAccount.objects.filter(pk=account.pk).exists() is keep_link
    assert SocialToken.objects.filter(pk=token.pk).exists() is keep_link
    assert SocialAccount.objects.filter(pk=other_account.pk).exists()
    oauth_user.refresh_from_db()
    emails = EmailAddress.objects.filter(user=oauth_user)
    if oauth_user.login_method == LoginMethod.PASSWORD:
        assert not emails.exists()
    else:
        email = emails.get()
        assert email.email == oauth_user.email
        assert email.primary and email.verified


@pytest.mark.django_db
def test_email_reassignment_denies_old_identity_and_allows_relink(
        admin_user, oauth_user, microsoft_login):
    _link(oauth_user)
    response = _save_user(admin_user, oauth_user, email='new@example.com')
    assert response.status_code == 200

    old_client, old_response = microsoft_login('person@example.com')
    _assert_denied(old_client, old_response, oauth_user)
    assert not SocialAccount.objects.filter(user=oauth_user).exists()

    new_client, new_response = microsoft_login('new@example.com', uid='new-microsoft-person')
    assert new_response.status_code == 302
    assert new_client.session[SESSION_KEY] == str(oauth_user.pk)
    assert SocialAccount.objects.get(user=oauth_user).uid == 'new-microsoft-person'


@pytest.mark.django_db
def test_policy_edit_rolls_back_links_and_whitelist_together(admin_user, oauth_user, monkeypatch):
    account = _link(oauth_user)

    def fail_mark_stale(*_args):
        raise RuntimeError('late failure')

    monkeypatch.setattr('apps.impostazioni.views.mark_stale', fail_mark_stale)

    response = _save_user(admin_user, oauth_user, email='new@example.com')

    assert response.status_code == 500
    oauth_user.refresh_from_db()
    assert oauth_user.email == 'person@example.com'
    assert SocialAccount.objects.filter(pk=account.pk, user=oauth_user).exists()
    assert EmailAddress.objects.get(user=oauth_user).email == oauth_user.email
    assert not UsedNonce.objects.filter(user=admin_user).exists()


@pytest.mark.django_db
def test_removing_a_link_alone_preserves_an_existing_session(oauth_user):
    account = _link(oauth_user)
    client = Client()
    client.force_login(oauth_user)
    account.delete()

    assert client.get('/prelievi').status_code == 200
    assert client.session[SESSION_KEY] == str(oauth_user.pk)


@pytest.mark.django_db
def test_deactivation_blocks_an_existing_session(oauth_user):
    client = Client()
    client.force_login(oauth_user)
    oauth_user.is_active = False
    oauth_user.save(update_fields=['is_active'])

    response = client.get('/prelievi')

    assert response.status_code == 302
    assert response.url.startswith(reverse('login'))
    assert not response.wsgi_request.user.is_authenticated
