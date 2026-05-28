/**
 * Per-page CSS lifecycle.  Each domain page imports its stylesheet at
 * mount and unhooks it at unmount, so styles don't leak across pages.
 */

export function loadCSS(url) {
  if (document.querySelector(`link[href="${url}"]`)) return;
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = url;
  document.head.appendChild(link);
}

export function unloadCSS(url) {
  document.querySelector(`link[href="${url}"]`)?.remove();
}
