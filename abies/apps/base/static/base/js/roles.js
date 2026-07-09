/**
 * Client-side role check.  Reads `data-role` on `<body>`, set server-side
 * from the authenticated user's role.  Authoritative gating still happens
 * in Django; this is for UI affordances only.
 */

import { ROLE_ADMIN, ROLE_WRITER } from './constants.js';

const WRITER_ROLES = [ROLE_ADMIN, ROLE_WRITER];

export function canModify() {
  return WRITER_ROLES.includes(document.body.dataset.role);
}
