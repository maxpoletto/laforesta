/**
 * Client-side router: pushState / popstate, page lifecycle.
 *
 * Each page module exports { mount(params), unmount(), onQueryChange(params) }.
 * The route table maps domain names to page modules.
 */

const PREFIX = '/abies/';

/** @type {Map<string, {mount: Function, unmount: Function, onQueryChange: Function}>} */
const routes = new Map();

let currentDomain = null;
let currentPage = null;
let contentEl = null;

// Per-domain last-visited URL.  Populated on tab click so that clicking a
// different tab restores its last state instead of resetting it.  Cleared
// on full page reload (module-level state).
const lastUrlPerDomain = new Map();

/**
 * Register a domain → page module.
 *
 * @param {string} domain — e.g. 'prelievi'
 * @param {{mount: Function, unmount: Function, onQueryChange: Function}} page
 */
export function addRoute(domain, page) {
  routes.set(domain, page);
}

/**
 * Parse the current URL into { domain, params }.
 */
function parseURL() {
  const path = location.pathname;
  const after = path.startsWith(PREFIX) ? path.slice(PREFIX.length) : '';
  const domain = after.split('/')[0] || null;
  const params = Object.fromEntries(new URLSearchParams(location.search));
  return { domain, params };
}

/**
 * Navigate to a URL.  Calls unmount/mount as needed.
 *
 * @param {string} url
 * @param {boolean} [replace=false] — use replaceState instead of pushState
 */
export function navigate(url, replace = false) {
  if (replace) {
    history.replaceState(null, '', url);
  } else {
    history.pushState(null, '', url);
  }
  render();
}

/**
 * Render the page for the current URL.
 */
function render() {
  const { domain, params } = parseURL();

  if (domain === currentDomain && currentPage) {
    currentPage.onQueryChange(params);
    updateActiveTab(domain);
    return;
  }

  if (currentPage) {
    currentPage.unmount();
  }

  currentDomain = domain;
  currentPage = routes.get(domain) || null;
  updateActiveTab(domain);

  if (currentPage) {
    currentPage.mount(params);
  } else {
    contentEl.replaceChildren();
  }
}

/**
 * Highlight the active tab in both desktop and mobile nav.
 */
function updateActiveTab(domain) {
  for (const el of document.querySelectorAll('.tab, .mobile-tab')) {
    el.classList.toggle('active', el.dataset.tab === domain);
  }
}

/**
 * Initialize the router.  Call once at boot.
 */
export function init() {
  contentEl = document.getElementById('content');

  // Intercept tab clicks for client-side navigation.
  for (const el of document.querySelectorAll('.tab, .mobile-tab')) {
    if (!el.dataset.tab) continue;
    el.addEventListener('click', (e) => {
      e.preventDefault();
      // Close mobile menu if open.
      document.getElementById('mobile-menu').classList.remove('open');

      const target = el.dataset.tab;
      // Stash current URL under current domain so that coming back here via
      // another tab click restores filter / sort / chart state.
      if (currentDomain) {
        lastUrlPerDomain.set(currentDomain, location.pathname + location.search);
      }
      // Switching to a different tab: restore its last URL if we have one.
      // Clicking the current tab: go to the bare href — i.e., reset state.
      const dest = target !== currentDomain && lastUrlPerDomain.has(target)
        ? lastUrlPerDomain.get(target)
        : el.getAttribute('href');
      navigate(dest);
    });
  }

  // Hamburger toggle.
  document.getElementById('hamburger').addEventListener('click', () => {
    document.getElementById('mobile-menu').classList.toggle('open');
  });

  // Close mobile menu on outside click.
  document.addEventListener('click', (e) => {
    const menu = document.getElementById('mobile-menu');
    const hamburger = document.getElementById('hamburger');
    if (menu.classList.contains('open') &&
        !menu.contains(e.target) &&
        !hamburger.contains(e.target)) {
      menu.classList.remove('open');
    }
  });

  // Back/forward buttons.
  window.addEventListener('popstate', render);

  // Initial render.
  render();
}
