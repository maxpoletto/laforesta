// Trigger a browser download of a Blob with a given filename. Single function;
// the indirection isolates the DOM call so app.js stays mostly UI-shaped.
'use strict';

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Release the object URL on next tick so the browser has time to start
  // the download. Some Android browsers race here.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadText(text, filename, mime) {
  const blob = new Blob([text], { type: mime || 'text/csv;charset=utf-8' });
  downloadBlob(blob, filename);
}
