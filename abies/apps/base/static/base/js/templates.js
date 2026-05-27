export function cloneTemplate(id) {
  const tmpl = document.getElementById(id);
  if (!tmpl) throw new Error(`Template #${id} not found`);
  return tmpl.content.cloneNode(true);
}
