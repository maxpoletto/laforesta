/** Audit times sort/export at full UTC precision and display in the site timezone. */
import { FIELD_TIMESTAMP_LABEL } from '../../base/js/constants.js';
import * as S from '../../base/js/strings.js';

const VALUE_COLUMN = { width: '420px', cellClassName: 'col-audit-value' };

export function auditColumnDefs(columns) {
  const labelIndex = columns.indexOf(FIELD_TIMESTAMP_LABEL);
  return {
    // String ordering preserves sub-millisecond precision; Date rounds it away.
    [S.COL_TIMESTAMP]: {
      label: S.COL_TIMESTAMP, type: 'string', width: '175px',
      formatter: (value, row) => row[labelIndex] ?? value,
    },
    [FIELD_TIMESTAMP_LABEL]: { hidden: true },
    [S.COL_USER]: { label: S.COL_USER, width: '150px' },
    [S.COL_TABLE]: { label: S.COL_TABLE, width: '170px' },
    [S.COL_ACTION]: { label: S.COL_ACTION, width: '120px' },
    [S.COL_OLD_VALUE]: { label: S.COL_OLD_VALUE, ...VALUE_COLUMN },
    [S.COL_NEW_VALUE]: { label: S.COL_NEW_VALUE, ...VALUE_COLUMN },
  };
}
