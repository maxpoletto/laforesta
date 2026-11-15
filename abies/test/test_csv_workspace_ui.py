"""Rendered workspaces expose data import separately from metadata editing."""
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest
from django.template.loader import render_to_string
from config import strings as S


class Node:
    def __init__(self, tag='', attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []

    def find(self, **attrs):
        if all(self.attrs.get(key) == value for key, value in attrs.items()):
            return self
        return next((found for child in self.children if (found := child.find(**attrs))), None)

    def tags(self, tag):
        return ([self] if self.tag == tag else []) + [node for child in self.children for node in child.tags(tag)]


class Tree(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.root = Node()
        self.stack = [self.root]
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in {'input', 'br', 'hr', 'img', 'meta', 'link'}:
            self.stack.append(node)

    def handle_endtag(self, tag):
        if len(self.stack) > 1:
            self.stack.pop()


def templates(app, modify=True):
    return Tree(render_to_string(f'{app}/_shell_templates_it.html',
        {'user': SimpleNamespace(can_modify=modify), 'S': S})).root


@pytest.mark.parametrize('app, identifier', [
    ('campionamenti', 'tmpl-campionamenti-edit-modal'),
    ('piano_di_taglio', 'tmpl-pdt-edit-plan'),
])
def test_pencil_dialogs_contain_only_metadata(app, identifier):
    modal = templates(app).find(id=identifier)
    assert modal is not None
    assert not any('modal-tabs' in node.attrs.get('class', '') for node in modal.tags('div'))
    assert not [node for node in modal.tags('input') if node.attrs.get('type') == 'file']
    assert modal.find(name='name') is not None
    assert modal.find(name='description') is not None


def test_sample_grid_actions_sit_with_its_summary():
    root = templates('campionamenti').find(id='tmpl-campionamenti-page')
    row = root.find(**{'data-target': 'grid-summary-row'})
    assert row is not None
    assert row.find(**{'data-target': 'grid-summary'}) is not None
    assert [node.attrs.get('data-action') for node in row.tags('button')] == [
        'export-grid-csv', 'import-grid-csv', 'add-area']


def test_sampled_trees_are_inside_the_survey_workspace():
    root = templates('campionamenti').find(id='tmpl-campionamenti-page')
    assert root.find(**{'data-section': 't'}) is None
    survey = root.find(**{'data-section': 'r', 'class': 'collapsible-body'})
    for target in ('survey-summary', 'survey-map-section', 'survey-map', 'trees-table-host', 'trees-detail-host'):
        assert survey.find(**{'data-target': target}) is not None, target


def test_read_only_grid_has_export_without_import_or_new_area():
    row = templates('campionamenti', False).find(**{'data-target': 'grid-summary-row'})
    assert row is not None
    assert [node.attrs.get('data-action') for node in row.tags('button')] == ['export-grid-csv']


def test_calendar_import_does_not_ask_for_a_forest_type():
    assert templates('piano_di_taglio').find(name='is_ceduo') is None


def test_survey_map_precedes_table_and_summary_with_an_area_filter_hint():
    survey = templates('campionamenti').find(**{'data-section':'r', 'class':'collapsible-body'})
    targets = [child.attrs.get('data-target') for child in survey.children]
    assert targets.index('survey-map-section') < targets.index('trees-table-host') < targets.index('trees-detail-host')
    map_section = survey.find(**{'data-target':'survey-map-section'})
    assert map_section.find(**{'data-field':'survey-map-title'}).tag == 'h3'
    hint = map_section.find(**{'data-target':'survey-map-help'})
    assert hint is not None and hint.attrs['class'] == 'subsection-note'
    assert survey.find(**{'data-target':'trees-scope-summary'}) is None
