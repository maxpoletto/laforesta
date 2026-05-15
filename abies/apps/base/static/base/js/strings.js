/**
 * App-wide string constants.
 *
 * Locale-specific strings live in `strings_<lang>.js` and are re-exported
 * below — change the import line to switch language.  Locale-INDEPENDENT
 * identifiers (wire markers, internal column names, etc.) are defined
 * directly in this file so they aren't duplicated across language files.
 */

export * from './strings_it.js';

// Internal identifiers — wire format, never displayed.
// Used both as digest-column keys and as JSON API field names.
export const ROW_ID  = 'row_id';
export const VERSION = 'version';

// JSON API field names shared between Python views and JS clients.
export const RECORD   = 'record';
export const RECORDS  = 'records';
export const DATA_ID  = 'data_id';
export const COLUMNS  = 'columns';
export const ROWS     = 'rows';
export const STATUS   = 'status';
export const MESSAGE  = 'message';
export const HTML     = 'html';

// JSON API `status` field values.
export const STATUS_CONFLICT         = 'conflict';
export const STATUS_VALIDATION_ERROR = 'validation_error';
export const STATUS_RATE_LIMITED     = 'rate_limited';
export const STATUS_NOT_FOUND        = 'not_found';
