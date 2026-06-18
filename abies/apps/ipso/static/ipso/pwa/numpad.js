// Custom on-screen numeric keypad. Big buttons, focus toggles between two
// fields (D and h). The HTML inputs themselves are inputmode="none"
// readonly so the native Android keyboard never appears.
//
// Why custom: native numeric keyboard on Android shifts layout when it
// opens and offers autocorrect for fast typers. The custom pad keeps the
// layout fixed and pipes every digit through one validated handler.
'use strict';

function createNumpad(opts) {
  // opts:
  //   container : DOM element to mount the buttons into
  //   inputs    : { <field>: HTMLInputElement, ... }
  //   onChange  : fn(field, value) called whenever any field changes
  //   maxLen    : { <field>: <digits>, ... } (default: 3 per field)
  const { container, inputs, onChange, maxLen } = opts;
  const max = maxLen || {};
  const fields = Object.keys(inputs);
  let focus = fields[0] || 'd';
  let buf = {};
  for (const k of fields) buf[k] = '';

  function value(field) { return buf[field]; }

  function setValue(field, v) {
    const s = String(v == null ? '' : v).slice(0, max[field] || 3);
    buf[field] = s;
    inputs[field].value = s;
    onChange && onChange(field, s);
  }

  function setFocus(field) {
    focus = field;
    for (const k of Object.keys(inputs)) {
      inputs[k].classList.toggle('focus', k === field);
    }
  }

  function press(ch) {
    const cur = buf[focus] || '';
    if (cur.length >= (max[focus] || 3)) return;
    // No leading zeros: '0' is rejected when empty; '07' rewrites to '7'.
    if (ch === '0' && cur.length === 0) return;
    setValue(focus, cur + ch);
  }

  function backspace() {
    const cur = buf[focus] || '';
    setValue(focus, cur.slice(0, -1));
  }

  function clear(field) {
    if (field) setValue(field, '');
    else { for (const k of fields) setValue(k, ''); }
  }

  function makeKey(label, handler, klass) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'np-key' + (klass ? ' ' + klass : '');
    b.textContent = label;
    b.addEventListener('click', (e) => {
      e.preventDefault();
      handler();
    });
    return b;
  }

  function mount() {
    container.replaceChildren();
    const grid = document.createElement('div');
    grid.className = 'np-grid';
    for (const d of ['1','2','3','4','5','6','7','8','9']) {
      grid.appendChild(makeKey(d, () => press(d)));
    }
    grid.appendChild(makeKey('⌫', backspace, 'np-back'));
    grid.appendChild(makeKey('0', () => press('0')));
    grid.appendChild(makeKey('C', () => clear(focus), 'np-clear'));
    container.appendChild(grid);

    // Tapping the inputs toggles focus through the numpad.
    for (const k of Object.keys(inputs)) {
      inputs[k].addEventListener('click', () => setFocus(k));
    }
    setFocus('d');
  }

  return {
    mount,
    setFocus,
    setValue,
    value,
    clear,
    getFocus() { return focus; },
  };
}
