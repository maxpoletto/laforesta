/** Small DOM fixture for exercising TableWrapper with the real vendored table. */
export class Element {
  constructor(tag = 'div', text = '') {
    this.tagName = tag.toLowerCase();
    this._text = text;
    this.children = [];
    this.parentNode = null;
    this.className = '';
    this.dataset = {};
    this.attributes = {};
    this.listeners = {};
    this.style = { setProperty: (key, value) => { this.style[key] = value; } };
    this.classList = {
      contains: key => this.className.split(/\s+/).includes(key),
      add: (...keys) => { this.className = [...new Set([...this.className.split(/\s+/), ...keys])].filter(Boolean).join(' '); },
      remove: (...keys) => { this.className = this.className.split(/\s+/).filter(key => !keys.includes(key)).join(' '); },
      toggle: (key, force) => {
        const active = force ?? !this.classList.contains(key);
        this.classList[active ? 'add' : 'remove'](key);
        return active;
      },
    };
  }
  get firstChild() { return this.children[0] || null; }
  get textContent() { return this.children.length ? this.children.map(c => c.textContent).join('') : this._text; }
  set textContent(value) { this.replaceChildren(); this._text = String(value ?? ''); }
  appendChild(child) {
    if (child.tagName === '#fragment') {
      for (const nested of [...child.children]) this.appendChild(nested);
      return child;
    }
    child.remove();
    child.parentNode = this;
    this.children.push(child);
    return child;
  }
  append(...children) { for (const child of children) this.appendChild(child); }
  replaceChildren(...children) {
    for (const child of this.children) child.parentNode = null;
    this.children = [];
    this._text = '';
    this.append(...children);
  }
  remove() {
    if (this.parentNode) this.parentNode.children = this.parentNode.children.filter(c => c !== this);
    this.parentNode = null;
  }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  getAttribute(key) {
    if (key.startsWith('data-')) return this.dataset[key.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] ?? null;
    return this.attributes[key] ?? this[key] ?? null;
  }
  removeAttribute(key) { delete this.attributes[key]; }
  matches(selector) {
    if (selector.includes(',')) return selector.split(',').some(part => this.matches(part.trim()));
    const attr = selector.match(/\[([^=\]]+)(?:="([^"]*)")?\]/);
    if (attr && (attr[2] === undefined ? this.getAttribute(attr[1]) == null : this.getAttribute(attr[1]) !== attr[2])) return false;
    const plain = selector.replace(/\[[^\]]+\]/g, '');
    const tag = plain.match(/^[a-z]+/i)?.[0];
    if (tag && tag !== this.tagName) return false;
    return [...plain.matchAll(/\.([\w-]+)/g)].every(match => this.classList.contains(match[1]));
  }
  closest(selector) { return this.matches(selector) ? this : this.parentNode?.closest(selector) || null; }
  querySelectorAll(selector) {
    const result = [];
    for (const child of this.children) {
      if (child.matches(selector)) result.push(child);
      result.push(...child.querySelectorAll(selector));
    }
    return result;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  contains(node) { return node === this || this.children.some(child => child.contains(node)); }
  cloneNode(deep = false) {
    const clone = new Element(this.tagName, this._text);
    clone.className = this.className;
    clone.dataset = { ...this.dataset };
    clone.attributes = { ...this.attributes };
    clone.style = { ...this.style };
    if (deep) for (const child of this.children) clone.appendChild(child.cloneNode(true));
    return clone;
  }
  addEventListener(event, fn) { (this.listeners[event] ||= []).push(fn); }
  removeEventListener(event, fn) { this.listeners[event] = (this.listeners[event] || []).filter(item => item !== fn); }
  focus() { this.focused = true; }
  async dispatch(type = 'click') {
    if (this.disabled) return;
    if (type === 'click' && this.type === 'checkbox') this.checked = !this.checked;
    const event = { target: this, stopped: false, preventDefault() {}, stopPropagation() { this.stopped = true; } };
    const pending = [];
    for (let node = this; node && !event.stopped; node = node.parentNode) {
      for (const fn of node.listeners[type] || []) pending.push(fn(event));
    }
    await Promise.all(pending);
  }
}

export function installDOM() {
  const body = new Element('body');
  body.dataset.csrf = 'test-csrf';
  globalThis.document = {
    body,
    createElement: tag => new Element(tag),
    createTextNode: text => new Element('#text', String(text)),
    createDocumentFragment: () => new Element('#fragment'),
    addEventListener() {}, removeEventListener() {},
    getElementById: id => body.querySelectorAll('[id]').find(el => el.id === id),
  };
  return body;
}
