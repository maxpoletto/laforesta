import { COL_COPPICE, ROW_ID, VERSION } from './constants.js';
import { fmtAccuracyM, fmtBool, fmtCoord, fmtDecimal1, fmtDecimal2, fmtDecimal3, fmtInt } from './format.js';
import * as S from './strings.js';

const labels = S.SAMPLED_TREE_COLUMN_LABELS;
const date = { label: S.LABEL_DATE, type: 'date', width: '90px' };
const number = { type: 'number', width: '80px', formatter: fmtInt };
const coordinates = { type: 'number', width: '80px', formatter: fmtCoord };

const shared = {
  [S.COL_PARCEL]: { label: S.COL_PARCEL, type: 'natural', searchMatch: 'exact', width: '85px' },
  [S.COL_SPECIES]: { label: S.COL_SPECIES, width: '120px' },
  [S.COL_D_CM]: { label: labels.diameter, type: 'number', width: '90px', formatter: fmtInt },
  [S.COL_H_M]: { label: labels.height, type: 'number', width: '70px', formatter: fmtDecimal2 },
  [S.COL_V_M3]: { label: labels.volume, type: 'number', width: '80px', formatter: fmtDecimal2 },
  [S.COL_MASS_Q]: { label: labels.mass, type: 'number', width: '90px', formatter: fmtDecimal1 },
  [S.COL_LAT]: { ...coordinates, label: S.COL_LAT },
  [S.COL_LON]: { ...coordinates, label: S.COL_LON },
  [S.COL_ACC_M]: { label: labels.accuracy, type: 'number', width: '75px', formatter: fmtAccuracyM },
};

export const SAMPLED_TREE_COLUMNS = {
  ...shared,
  [S.COL_SAMPLE_DATE]: date,
  [S.COL_REGION]: { label: S.COL_REGION, searchMatch: 'exact', width: '90px' },
  [S.COL_AREA_NUM]: { label: labels.area, type: 'natural', searchMatch: 'exact', width: '55px' },
  [S.COL_TREE_NUM]: { ...number, label: labels.number },
  [S.COL_TYPE]: { label: S.COL_TYPE, width: '65px' },
  [S.COL_COPPICE_SHOOT]: { label: labels.shoot, type: 'number', width: '55px', formatter: fmtInt },
  [S.COL_COPPICE_STD]: { label: labels.standard, type: 'boolean', width: '55px', formatter: fmtBool },
  [S.COL_L10_MM]: { label: S.COL_L10_MM, type: 'number', width: '85px', formatter: fmtInt },
  [S.COL_PRESERVED]: { label: S.COL_PRESERVED, type: 'boolean', width: '50px', formatter: fmtBool },
  [S.COL_SAMPLE_AREA]: { hidden: true },
  [COL_COPPICE]: { hidden: true },
  [S.COL_PRESSLER]: { hidden: true },
  [VERSION]: { hidden: true },
};

const withDash = format => value => value == null || value === '' ? '-' : format(value);
const decimal2 = withDash(fmtDecimal2);
const marked = {
  ...shared,
  [S.COL_DATE]: date,
  [S.COL_NUMBER]: number,
  [S.COL_H_MEASURED]: { label: S.MARKED_TREE_COLUMN_LABELS.heightMeasured, type: 'boolean', width: '90px' },
  [S.COL_H_M]: { ...shared[S.COL_H_M], formatter: decimal2 },
  [S.COL_V_M3]: { ...shared[S.COL_V_M3], formatter: withDash(fmtDecimal3) },
  [S.COL_MASS_Q]: { ...shared[S.COL_MASS_Q], formatter: decimal2 },
  [S.COL_OPERATOR]: { width: '100px' },
  [VERSION]: { hidden: true },
};

export function buildMarkTreeColumnDefs(columns, showParcel) {
  return Object.fromEntries(columns.filter(name => name !== ROW_ID).map(name => [name, {
    label: name,
    ...(name === S.COL_PARCEL && !showParcel ? { hidden: true } : marked[name]),
  }]));
}
