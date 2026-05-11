"""Schema-level enforcement of the sample-belongs-to-survey-grid invariant.

A `sample` row's `sample_area.sample_grid_id` MUST equal its
`survey.sample_grid_id`.  Documented in `docs/database.md` under
"Surveys and samples".  Enforced at the schema level via SQLite
triggers (CHECK constraints can't reference other tables in SQLite).
"""

from django.db import migrations


TRIGGERS_SQL = """
CREATE TRIGGER sample_grid_match_insert BEFORE INSERT ON base_sample
BEGIN
    SELECT RAISE(ABORT, 'sample.sample_area must belong to survey.sample_grid')
    WHERE (SELECT sample_grid_id FROM base_samplearea WHERE id = NEW.sample_area_id)
       <> (SELECT sample_grid_id FROM base_survey     WHERE id = NEW.survey_id);
END;

CREATE TRIGGER sample_grid_match_update BEFORE UPDATE ON base_sample
BEGIN
    SELECT RAISE(ABORT, 'sample.sample_area must belong to survey.sample_grid')
    WHERE (SELECT sample_grid_id FROM base_samplearea WHERE id = NEW.sample_area_id)
       <> (SELECT sample_grid_id FROM base_survey     WHERE id = NEW.survey_id);
END;
"""

TRIGGERS_REVERSE = """
DROP TRIGGER IF EXISTS sample_grid_match_insert;
DROP TRIGGER IF EXISTS sample_grid_match_update;
"""


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0001_initial'),
    ]

    operations = [
        migrations.RunSQL(sql=TRIGGERS_SQL, reverse_sql=TRIGGERS_REVERSE),
    ]
