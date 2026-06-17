// CSV serialisation for ipso, matching abies' Italian conventions:
// semicolon separator, comma decimal, UTF-8 BOM, CRLF newlines,
// DD/MM/YYYY date format.
//
// Pure functions only — testable from node, no DOM, no Date timezone surprises
// (caller provides a Date for filename and a YYYY-MM-DD string for the row).
'use strict';

const CSV_BOM = '﻿';
const CSV_SEP = ';';
const CSV_NL = '\r\n';
const HEADER = [
  'Data', 'Compresa', 'Particella', 'Catastrofata',
  'Numero', 'Specie',
  'D_cm', 'H_m', 'H_measured',
  'Lat', 'Lon', 'Acc_m',
  'Operatore',
];

// Sentinel used in the filename's particella slot for catastrofate sessions.
const FILENAME_CATASTROFATE = 'catastrofate';

function pad2(n) { return n < 10 ? '0' + n : '' + n; }

// Convert YYYY-MM-DD (the form session.data holds) to DD/MM/YYYY.
function formatDate(iso) {
  const m = String(iso).match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!m) throw new Error('formatDate: expected YYYY-MM-DD, got ' + iso);
  return m[3] + '/' + m[2] + '/' + m[1];
}

// Italian decimal: dot -> comma. Pass-through for integers and empty values.
function fmtFloat(v, decimals) {
  if (v == null || Number.isNaN(v)) return '';
  return v.toFixed(decimals).replace('.', ',');
}

function fmtInt(v) {
  if (v == null || Number.isNaN(v)) return '';
  return '' + Math.round(v);
}

// Defensive escape: wrap a field in double quotes if it contains the
// separator, a double quote, or a newline. Double up internal quotes.
function escapeField(s) {
  const str = String(s == null ? '' : s);
  if (/[;"\r\n]/.test(str)) {
    return '"' + str.replace(/"/g, '""') + '"';
  }
  return str;
}

function formatHeader() {
  return HEADER.map(escapeField).join(CSV_SEP);
}

function formatRow(rec, session) {
  const catastrofata = !!session.catastrofata;
  // Particella is per-tree (auto-detected or manually overridden) so it
  // is carried on `rec`, not the session. Catastrofate sessions are
  // identified by the dedicated column; their Particella column is the
  // actual mark location, not blank.
  const cells = [
    formatDate(session.data),
    session.compresa,
    rec.particella || '',
    catastrofata ? '1' : '0',
    fmtInt(rec.numero),
    rec.specie,
    fmtInt(rec.d_cm),
    fmtInt(rec.h_m),
    rec.h_measured ? '1' : '0',
    fmtFloat(rec.lat, 6),
    fmtFloat(rec.lng, 6),
    fmtInt(rec.acc_m),
    session.operatore || '',
  ];
  return cells.map(escapeField).join(CSV_SEP);
}

// Full CSV file as a string. Caller wraps in a Blob and triggers download.
function formatFile(session, trees) {
  const lines = [formatHeader()];
  for (const t of trees) lines.push(formatRow(t, session));
  return CSV_BOM + lines.join(CSV_NL) + CSV_NL;
}

// Filename: ipso_<compresa>[_catastrofate]_<YYYY-MM-DD>_<HHMM>.csv
// Particella is no longer carried on the session (it's per-tree now), so
// it does not appear in the filename. The HHMM stamp keeps multiple
// sessions in the same compresa on the same day distinct.
// `now` is a Date (caller supplies; tests inject a fixed Date).
// `kind` is 'final' (default) or 'backup' (then includes a seq suffix).
function filename(session, now, kind, seq) {
  const compresa = sanitize(session.compresa);
  const date = session.data;  // already YYYY-MM-DD
  const hhmm = pad2(now.getHours()) + pad2(now.getMinutes());
  const parts = ['ipso', compresa];
  if (session.catastrofata) parts.push(FILENAME_CATASTROFATE);
  parts.push(date, hhmm);
  let base = parts.join('_');
  if (kind === 'backup') base += '_backup_' + seq;
  return base + '.csv';
}

// Replace anything that's not [A-Za-z0-9._-] with underscore. Keeps the
// filename safe across Android Downloads providers.
function sanitize(s) {
  return String(s).replace(/[^A-Za-z0-9._-]+/g, '_');
}

const csv = {
  CSV_BOM, CSV_SEP, CSV_NL, HEADER, FILENAME_CATASTROFATE,
  formatDate, fmtFloat, fmtInt, escapeField,
  formatHeader, formatRow, formatFile, filename, sanitize,
};
if (typeof module !== 'undefined') module.exports = csv;
