/**
 * Escape-key handling.  Page modules use this to dismiss a sub-view
 * (e.g. close item view, return to table) when the user presses Esc.
 *
 * The handler always defers to an open modal — if `#modal-container.open`
 * is set, the modal's own Esc dismissal takes precedence and the
 * page-level handler does nothing.
 */

export function installEscapeHandler(onEscape) {
  const handler = (e) => {
    if (e.key !== 'Escape') return;
    if (document.getElementById('modal-container')?.classList.contains('open')) return;
    onEscape();
  };
  document.addEventListener('keydown', handler);
  return () => document.removeEventListener('keydown', handler);
}
