#!/usr/bin/env node
/// Executable regression harness for the `/play` startup reconciliation (`reconcileStartup`).
///
/// Runs the REAL script extracted from the served `play.html` inside a Node `vm` context with a
/// stubbed browser environment (DOM elements, `history`, `location`, `localStorage` and a small
/// functional in-memory IndexedDB fake). Each scenario seeds the fake IndexedDB with saved tabs,
/// executes the page script (which calls `reconcileStartup` at top level), waits for the trailing
/// debounced `persist` to write the reconciled workspace back, and then asserts both the live
/// `tabs` state and what was persisted.
///
/// Driven by `test.py` inside the `clickhouse/mysql-js-client` container (node:22-alpine),
/// against the `/play` page served by a real ClickHouse server. Can also be run standalone
/// against a checkout for development: node reconcile_harness.js programs/server/play.html
///
/// Usage: node reconcile_harness.js <path-or-url-of-play.html>
/// Exit code 0 = all scenarios pass; 1 = failure (details on stdout).

'use strict';

const vm = require('vm');
const fs = require('fs');

/// ----- Fake DOM -----------------------------------------------------------------

function makeStyle() {
    return new Proxy({
        setProperty() {},
        removeProperty() {},
        getPropertyValue() { return ''; },
    }, {
        get(target, prop) {
            if (prop in target) return target[prop];
            return '';
        },
        set(target, prop, value) { target[prop] = value; return true; },
    });
}

function makeClassList() {
    const set = new Set();
    return {
        add(...cs) { for (const c of cs) set.add(c); },
        remove(...cs) { for (const c of cs) set.delete(c); },
        toggle(c, force) {
            const on = force === undefined ? !set.has(c) : !!force;
            if (on) set.add(c); else set.delete(c);
            return on;
        },
        contains(c) { return set.has(c); },
    };
}

function makeElement(tag) {
    const listeners = new Map();
    const attributes = new Map();
    const el = {
        tagName: String(tag || 'div').toUpperCase(),
        nodeType: 1,
        id: '',
        style: makeStyle(),
        classList: makeClassList(),
        dataset: {},
        children: [],
        childNodes: [],
        parentNode: null,
        parentElement: null,
        firstChild: null,
        lastChild: null,
        nextSibling: null,
        previousSibling: null,
        value: '',
        textContent: '',
        innerHTML: '',
        innerText: '',
        title: '',
        placeholder: '',
        className: '',
        name: '',
        type: '',
        href: '',
        hidden: false,
        disabled: false,
        checked: false,
        readOnly: false,
        contentEditable: 'inherit',
        spellcheck: true,
        tabIndex: 0,
        selectionStart: 0,
        selectionEnd: 0,
        selectionDirection: 'none',
        scrollTop: 0,
        scrollLeft: 0,
        scrollHeight: 0,
        scrollWidth: 0,
        clientHeight: 0,
        clientWidth: 0,
        offsetHeight: 0,
        offsetWidth: 0,
        offsetTop: 0,
        offsetLeft: 0,

        addEventListener(type, fn) {
            if (!listeners.has(type)) listeners.set(type, []);
            listeners.get(type).push(fn);
        },
        removeEventListener(type, fn) {
            const l = listeners.get(type);
            if (l) {
                const i = l.indexOf(fn);
                if (i !== -1) l.splice(i, 1);
            }
        },
        dispatchEvent(ev) {
            try {
                Object.defineProperty(ev, 'target', { value: el, configurable: true });
                Object.defineProperty(ev, 'currentTarget', { value: el, configurable: true });
            } catch (e) { /* already defined */ }
            for (const fn of (listeners.get(ev.type) || []).slice()) fn.call(el, ev);
            const handler = el['on' + ev.type];
            if (typeof handler === 'function') handler.call(el, ev);
            return true;
        },
        appendChild(c) {
            el.children.push(c);
            el.childNodes.push(c);
            c.parentNode = el;
            c.parentElement = el;
            el.firstChild = el.children[0];
            el.lastChild = c;
            return c;
        },
        removeChild(c) {
            el.children = el.children.filter(x => x !== c);
            el.childNodes = el.childNodes.filter(x => x !== c);
            el.firstChild = el.children[0] || null;
            el.lastChild = el.children[el.children.length - 1] || null;
            return c;
        },
        insertBefore(c, ref) {
            const i = el.children.indexOf(ref);
            if (i === -1) return el.appendChild(c);
            el.children.splice(i, 0, c);
            el.childNodes.splice(i, 0, c);
            c.parentNode = el;
            c.parentElement = el;
            el.firstChild = el.children[0];
            return c;
        },
        replaceChildren(...cs) {
            el.children = [...cs];
            el.childNodes = [...cs];
            for (const c of cs) { c.parentNode = el; c.parentElement = el; }
            el.firstChild = el.children[0] || null;
            el.lastChild = el.children[el.children.length - 1] || null;
        },
        remove() { if (el.parentNode) el.parentNode.removeChild(el); },
        setAttribute(k, v) { attributes.set(k, String(v)); if (k === 'id') el.id = String(v); },
        getAttribute(k) { return attributes.has(k) ? attributes.get(k) : null; },
        removeAttribute(k) { attributes.delete(k); },
        hasAttribute(k) { return attributes.has(k); },
        focus() {},
        blur() {},
        click() { el.dispatchEvent(new Event('click')); },
        select() {},
        setSelectionRange(a, b) { el.selectionStart = a; el.selectionEnd = b; },
        getBoundingClientRect() { return { top: 0, left: 0, right: 0, bottom: 0, width: 0, height: 0, x: 0, y: 0 }; },
        getClientRects() { return []; },
        querySelector() { return null; },
        querySelectorAll() { return []; },
        closest() { return null; },
        matches() { return false; },
        contains() { return false; },
        scrollIntoView() {},
        scrollTo() {},
        scroll() {},
        cloneNode() { return makeElement(el.tagName); },
        insertAdjacentElement() {},
        insertAdjacentHTML() {},
        insertAdjacentText() {},
        getContext() { return null; },
        /// Methods of the <query-result> / <query-progress> custom elements: with a stub DOM the
        /// custom-element upgrade never happens, so provide inert versions of everything the
        /// script calls on them. The seeded run-backed snapshot carries no `data`, so
        /// `restoreFromHistory` bails out before any real rendering.
        clear() {},
        update() { return true; },
        updateRaw() {},
        renderError() {},
        clearError() {},
        clearSelection() {},
        flushFragment() {},
        async renderChart() {},
        redrawChart() {},
        renderGraph() {},
        renderTotals() {},
        applyColumnColors() {},
        refreshColumnColor() {},
        transposeIfNeeded() {},
        _changeTableLayout() {},
        start() {},
        finish() {},
        updateProgress() {},
        updateText() {},
        attachShadow() { return makeElement('shadow-root'); },
    };
    return el;
}

function makeDocument() {
    const byId = new Map();
    const doc = makeElement('#document');
    doc.nodeType = 9;
    doc.readyState = 'complete';
    doc.visibilityState = 'visible';
    doc.hidden = false;
    doc.cookie = '';
    doc.body = makeElement('body');
    doc.head = makeElement('head');
    doc.documentElement = makeElement('html');
    doc.activeElement = doc.body;
    doc.getElementById = (id) => {
        if (!byId.has(id)) {
            const el = makeElement('div');
            el.id = id;
            byId.set(id, el);
        }
        return byId.get(id);
    };
    doc.createElement = (tag) => makeElement(tag);
    doc.createElementNS = (ns, tag) => makeElement(tag);
    doc.createTextNode = (text) => {
        const el = makeElement('#text');
        el.nodeType = 3;
        el.textContent = String(text);
        return el;
    };
    doc.createDocumentFragment = () => makeElement('#document-fragment');
    doc.createRange = () => ({
        selectNodeContents() {},
        setStart() {},
        setEnd() {},
        collapse() {},
        cloneRange() { return this; },
        getBoundingClientRect() { return { top: 0, left: 0, right: 0, bottom: 0, width: 0, height: 0 }; },
        getClientRects() { return []; },
    });
    doc.execCommand = () => false;
    doc.queryCommandSupported = () => false;
    const bySelector = new Map();
    doc.querySelector = (sel) => {
        if (!bySelector.has(sel)) bySelector.set(sel, makeElement('div'));
        return bySelector.get(sel);
    };
    doc.querySelectorAll = () => [];
    doc.hasFocus = () => true;
    /// The favicon <link> carries a base64 SVG data URL that the script recolors at load.
    doc.querySelector('link[rel="icon"]').href =
        'data:image/svg+xml;base64,' + Buffer.from('<svg fill="#ff0"></svg>').toString('base64');
    return doc;
}

/// ----- Fake IndexedDB (only what `openDb`/`loadFromDb`/`persist` use) --------------------

function makeIndexedDB(seedTabs, seedMeta, openDelayMs) {
    const stores = new Map();
    stores.set('tabs', { keyPath: 'id', data: new Map((seedTabs || []).map(r => [r.id, structuredClone(r)])) });
    stores.set('meta', { keyPath: 'key', data: new Map(seedMeta ? [['state', structuredClone(seedMeta)]] : []) });
    const stats = { persistCount: 0 };

    function makeStoreHandle(name) {
        const s = stores.get(name);
        return {
            getAll() { return { result: [...s.data.values()].map(v => structuredClone(v)) }; },
            get(key) {
                const v = s.data.get(key);
                return { result: v === undefined ? undefined : structuredClone(v) };
            },
            put(obj) {
                s.data.set(obj[s.keyPath], structuredClone(obj));
                /// `persist` writes the meta `state` record last; count completed workspace saves.
                if (name === 'meta' && obj.key === 'state') stats.persistCount++;
                return { result: obj[s.keyPath] };
            },
            clear() { s.data.clear(); return { result: undefined }; },
            delete(key) { s.data.delete(key); return { result: undefined }; },
        };
    }

    const indexedDB = {
        open(name, version) {
            const req = { onupgradeneeded: null, onsuccess: null, onerror: null, result: null };
            /// `openDelayMs` lets a scenario make `IndexedDB.open` slower than any auto-run that
            /// races startup reconciliation (see the stale-reload-run-race scenario).
            setTimeout(() => {
                req.result = {
                    objectStoreNames: { contains: (n) => stores.has(n) },
                    createObjectStore(n, opts) {
                        if (!stores.has(n)) stores.set(n, { keyPath: opts.keyPath, data: new Map() });
                        return makeStoreHandle(n);
                    },
                    transaction(names, mode) {
                        const tx = { oncomplete: null, onerror: null, onabort: null };
                        tx.objectStore = (n) => makeStoreHandle(n);
                        setTimeout(() => { if (tx.oncomplete) tx.oncomplete(); }, 0);
                        return tx;
                    },
                    close() {},
                };
                if (req.onsuccess) req.onsuccess();
            }, openDelayMs || 0);
            return req;
        },
    };
    return { indexedDB, stores, stats };
}

/// ----- Other browser globals ------------------------------------------------------------

function makeStorage() {
    const map = new Map();
    return {
        getItem(k) { return map.has(k) ? map.get(k) : null; },
        setItem(k, v) { map.set(String(k), String(v)); },
        removeItem(k) { map.delete(k); },
        clear() { map.clear(); },
        key(i) { return [...map.keys()][i] ?? null; },
        get length() { return map.size; },
    };
}

function makeLocation(href) {
    const u = new URL(href);
    return {
        get href() { return u.href; },
        get origin() { return u.origin; },
        get protocol() { return u.protocol; },
        get host() { return u.host; },
        get hostname() { return u.hostname; },
        get port() { return u.port; },
        get pathname() { return u.pathname; },
        get search() { return u.search; },
        get hash() { return u.hash; },
        set hash(h) { u.hash = h; },
        toString() { return u.href; },
        assign() {},
        replace() {},
        reload() {},
        _apply(url) {
            const next = new URL(url, u.href);
            u.href = next.href;
        },
    };
}

function makeHistory(initialState, location) {
    return {
        state: initialState,
        length: 1,
        replaceState(state, title, url) {
            this.state = state;
            if (url !== undefined && url !== null) location._apply(String(url));
        },
        pushState(state, title, url) {
            this.state = state;
            this.length++;
            if (url !== undefined && url !== null) location._apply(String(url));
        },
        back() {},
        forward() {},
        go() {},
    };
}

/// ----- Context assembly -------------------------------------------------------------------

function makeContext({ href, historyState, seedTabs, seedMeta, openDelayMs }) {
    const document = makeDocument();
    const location = makeLocation(href);
    const history = makeHistory(historyState, location);
    const { indexedDB, stores, stats } = makeIndexedDB(seedTabs, seedMeta, openDelayMs);

    const sandbox = {
        document,
        location,
        history,
        indexedDB,
        localStorage: makeStorage(),
        sessionStorage: makeStorage(),
        navigator: {
            clipboard: { writeText: async () => {}, readText: async () => '' },
            platform: 'Linux x86_64',
            language: 'en-US',
            userAgent: 'play-reconcile-harness',
        },
        /// Deterministic environment: no network. The only top-level fetch (the webterminal
        /// probe) checks `resp.ok`, and every other call site handles a non-ok response.
        fetch: async () => ({
            ok: false,
            status: 503,
            statusText: 'harness: network disabled',
            headers: { get: () => null },
            text: async () => '',
            json: async () => ({}),
        }),
        setTimeout, clearTimeout, setInterval, clearInterval,
        queueMicrotask,
        requestAnimationFrame: (fn) => setTimeout(fn, 0),
        cancelAnimationFrame: (t) => clearTimeout(t),
        requestIdleCallback: (fn) => setTimeout(fn, 0),
        cancelIdleCallback: (t) => clearTimeout(t),
        console,
        performance: { now: () => Date.now() },
        atob: (b64) => Buffer.from(b64, 'base64').toString('binary'),
        btoa: (bin) => Buffer.from(bin, 'binary').toString('base64'),
        TextEncoder, TextDecoder,
        URL, URLSearchParams,
        Event, CustomEvent,
        AbortController,
        structuredClone,
        HTMLElement: class HTMLElement {},
        customElements: { define() {}, get() { return undefined; }, whenDefined() { return Promise.resolve(); } },
        ResizeObserver: class ResizeObserver { observe() {} unobserve() {} disconnect() {} },
        MutationObserver: class MutationObserver { observe() {} disconnect() {} takeRecords() { return []; } },
        IntersectionObserver: class IntersectionObserver { observe() {} unobserve() {} disconnect() {} },
        matchMedia: () => ({ matches: false, media: '', addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} }),
        getComputedStyle: () => new Proxy({ getPropertyValue: () => '' }, { get(t, p) { return p in t ? t[p] : ''; } }),
        getSelection: () => ({ removeAllRanges() {}, addRange() {}, toString() { return ''; }, rangeCount: 0 }),
        alert() {}, confirm() { return false; }, prompt() { return null; },
        scrollTo() {}, scroll() {},
        innerHeight: 800, innerWidth: 1280, devicePixelRatio: 1,
        addEventListener() {}, removeEventListener() {},
        WebAssembly,
    };
    sandbox.window = sandbox;
    sandbox.self = sandbox;
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    return { sandbox, stores, stats };
}

/// ----- Scenario driver ----------------------------------------------------------------------

function extractScript(html) {
    const blocks = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
    if (!blocks.length) throw new Error('no <script> block found in play.html');
    return blocks.reduce((a, b) => (a.length >= b.length ? a : b));
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function runScenario(js, config) {
    const { sandbox, stores, stats } = makeContext(config);
    vm.runInContext(js, sandbox, { filename: 'play.html.js' });
    /// Startup is asynchronous: `reconcileStartup` awaits IndexedDB and ends with the debounced
    /// `scheduleSave` (400 ms), whose `persist` writes the reconciled workspace back. Wait for
    /// that write — it marks reconciliation as complete and persisted.
    const deadline = Date.now() + 15000;
    while (stats.persistCount < 1) {
        if (Date.now() > deadline) throw new Error('timed out waiting for the startup persist');
        await sleep(25);
    }
    await sleep(50);
    const live = vm.runInContext(
        'JSON.stringify({ tabs: tabs.map(t => ({ id: t.id, title: t.title, query: t.query, ran: !!(t.result && t.result.ran) })), activeTabId })',
        sandbox);
    const persisted = [...stores.get('tabs').data.values()];
    return { live: JSON.parse(live), persisted, sandbox };
}

/// ----- Assertions ----------------------------------------------------------------------------

let failures = 0;

function check(scenario, what, cond, actual) {
    if (cond) {
        console.log(`PASS [${scenario}] ${what}`);
    } else {
        failures++;
        console.log(`FAIL [${scenario}] ${what} -- actual: ${JSON.stringify(actual)}`);
    }
}

async function main() {
    const src = process.argv[2];
    if (!src) {
        console.error('usage: node reconcile_harness.js <path-or-url-of-play.html>');
        process.exit(2);
    }
    let html;
    if (/^https?:/.test(src)) {
        const resp = await fetch(src);
        if (!resp.ok) throw new Error(`GET ${src} -> HTTP ${resp.status}`);
        html = await resp.text();
    } else {
        html = fs.readFileSync(src, 'utf8');
    }
    const js = extractScript(html);
    const base = 'http://127.0.0.1:8123/play';

    /// Contract 1: a mixed workspace (blank + non-blank saved tabs) restores only the
    /// non-blank tabs on a plain load; the blank one is pruned from IndexedDB too.
    {
        const r = await runScenario(js, {
            href: base,
            historyState: null,
            seedTabs: [
                { id: 't7', title: 'Scratch', query: '   \n  ', params: {}, result: null, lastSavedQuery: '' },
                { id: 't8', title: 'Report', query: 'SELECT 1', params: {}, result: null, lastSavedQuery: 'SELECT 1' },
            ],
            seedMeta: { key: 'state', activeTabId: 't8', tabOrder: ['t7', 't8'], tabSeq: 8, tabTitleSeq: 2 },
        });
        check('mixed', 'only the non-blank tab is restored',
            r.live.tabs.length === 1 && r.live.tabs[0].title === 'Report' && r.live.tabs[0].query === 'SELECT 1',
            r.live);
        check('mixed', 'the blank tab is pruned from IndexedDB',
            r.persisted.length === 1 && r.persisted[0].id === 't8',
            r.persisted.map(p => p.id));
    }

    /// Contract 2: an all-blank workspace falls back to a single fresh tab,
    /// exactly as on a first-ever visit.
    {
        const r = await runScenario(js, {
            href: base,
            historyState: null,
            seedTabs: [
                { id: 't7', title: 'Scratch', query: '', params: {}, result: null, lastSavedQuery: '' },
                { id: 't8', title: 'Notes', query: ' \t ', params: {}, result: null, lastSavedQuery: '' },
            ],
            seedMeta: { key: 'state', activeTabId: 't7', tabOrder: ['t7', 't8'], tabSeq: 8, tabTitleSeq: 2 },
        });
        check('all-blank', 'exactly one fresh empty tab remains',
            r.live.tabs.length === 1 && r.live.tabs[0].query.trim() === '',
            r.live);
        check('all-blank', 'neither blank record survives in IndexedDB',
            !r.persisted.some(p => p.title === 'Scratch' || p.title === 'Notes'),
            r.persisted.map(p => p.title));
    }

    /// Contract 3: a tab whose editor was cleared after a run still holds a `result.ran`
    /// snapshot and must be preserved, not pruned.
    {
        const r = await runScenario(js, {
            href: base,
            historyState: null,
            seedTabs: [
                { id: 't7', title: 'Ran', query: '', params: {}, result: { ran: true, query: 'SELECT 2', params: {} }, lastSavedQuery: 'SELECT 2' },
                { id: 't8', title: 'Scratch', query: '', params: {}, result: null, lastSavedQuery: '' },
            ],
            seedMeta: { key: 'state', activeTabId: 't7', tabOrder: ['t7', 't8'], tabSeq: 8, tabTitleSeq: 2 },
        });
        check('run-backed', 'the cleared-after-run tab survives, the blank one is pruned',
            r.live.tabs.length === 1 && r.live.tabs[0].title === 'Ran' && r.live.tabs[0].ran,
            r.live);
        check('run-backed', 'the run-backed record stays in IndexedDB',
            r.persisted.some(p => p.id === 't7' && p.result && p.result.ran) && !r.persisted.some(p => p.id === 't8'),
            r.persisted.map(p => p.id));
    }

    /// Guard: a plain reload whose URL still carries `?tab=<pruned blank tab>` (a stale echo:
    /// `history.state` was preserved by the reload and names the pruned tab) must NOT resurrect
    /// the blank tab; it falls back to a surviving saved tab.
    {
        const r = await runScenario(js, {
            href: base + '?tab=Scratch',
            historyState: { tabId: 't7', tabName: 'Scratch' },
            seedTabs: [
                { id: 't7', title: 'Scratch', query: '', params: {}, result: null, lastSavedQuery: '' },
                { id: 't8', title: 'Report', query: 'SELECT 1', params: {}, result: null, lastSavedQuery: 'SELECT 1' },
            ],
            seedMeta: { key: 'state', activeTabId: 't7', tabOrder: ['t7', 't8'], tabSeq: 8, tabTitleSeq: 2 },
        });
        check('stale-reload', 'the pruned blank tab is not resurrected; the survivor is restored',
            r.live.tabs.length === 1 && r.live.tabs[0].title === 'Report',
            r.live);
    }

    /// Guard (startup race): a bare stale `?tab=<pruned blank>&run=1` reload must still fall back
    /// to the survivor even when `IndexedDB.open` is slower than the auto-run. `run=1` carries no
    /// `#<query>` hash here, so the auto-run must NOT fire at the top level before reconciliation:
    /// if it did, its `postAll`/`saveHistory` would rewrite `history.state` to the bootstrap tab
    /// while `loadFromDb` was still opening, `stale_blank_reload` would stop matching the pruned
    /// blank tab, and the authoritative path would recreate and re-persist `Scratch`. Delaying
    /// `IndexedDB.open` by 30 ms makes that race deterministic. With the fix (every startup `run=1`
    /// deferred to `reconcileStartup`), the workspace still falls back to `Report` and `Scratch`
    /// stays pruned.
    {
        const r = await runScenario(js, {
            href: base + '?tab=Scratch&run=1',
            historyState: { tabId: 't7', tabName: 'Scratch' },
            openDelayMs: 30,
            seedTabs: [
                { id: 't7', title: 'Scratch', query: '', params: {}, result: null, lastSavedQuery: '' },
                { id: 't8', title: 'Report', query: 'SELECT 1', params: {}, result: { ran: true, query: 'SELECT 1', params: {} }, lastSavedQuery: 'SELECT 1' },
            ],
            seedMeta: { key: 'state', activeTabId: 't7', tabOrder: ['t7', 't8'], tabSeq: 8, tabTitleSeq: 2 },
        });
        check('stale-reload-run-race', 'the blank tab is not resurrected under a slow IndexedDB open',
            r.live.tabs.length === 1 && r.live.tabs[0].title === 'Report',
            r.live);
        check('stale-reload-run-race', 'no blank Scratch is re-persisted to IndexedDB',
            !r.persisted.some(p => p.title === 'Scratch'),
            r.persisted.map(p => p.title));
    }

    if (failures) {
        console.log(`${failures} check(s) FAILED`);
        process.exit(1);
    }
    console.log('All scenarios passed');
}

main().catch((e) => {
    console.log('HARNESS ERROR: ' + (e && e.stack || e));
    process.exit(1);
});
