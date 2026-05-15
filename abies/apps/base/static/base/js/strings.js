/**
 * App-wide string constants.
 *
 * Locale-specific strings live in `strings_<lang>.js` and are re-exported
 * below — change the import line to switch language.  Locale-INDEPENDENT
 * identifiers (wire markers, internal column names, etc.) are defined
 * directly in this file so they aren't duplicated across language files.
 */

export * from './strings_it.js';

// Internal digest column identifiers — wire format, never displayed.
export const COL_ROW_ID  = 'row_id';
export const COL_VERSION = 'version';
