const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const NOW = 1700000000000;
const ORIGINAL_CLOUD_LINK =
  'https://console.clickhouse.cloud/signUp?loc=docs-nav-signUp-cta';
const SCRIPT = fs.readFileSync(path.join(__dirname, 'galaxy.js'), 'utf8');
// Snapshot of the request/link contract produced by clickhouse-docs'
// GalaxyClient, useGalaxyOnPage, and utmPersistence implementations.
const LEGACY_CONTRACT = JSON.parse(fs.readFileSync(
  path.join(__dirname, 'fixtures', 'galaxy-legacy-contract.json'),
  'utf8',
));

function createStorage() {
  const values = new Map();
  return {
    getItem(key) {
      return values.has(key) ? values.get(key) : null;
    },
    setItem(key, value) {
      values.set(key, String(value));
    },
    removeItem(key) {
      values.delete(key);
    },
  };
}

function createEnvironment(pathname, search = '', origin = 'https://clickhouse.com') {
  const location = new URL(`${origin}${pathname}${search}`);
  const cookies = new Map([['_ga', 'GA1.1.123.456']]);
  const listeners = new Map();
  const beacons = [];
  const clearedIntervals = [];
  const animationFrames = [];
  const cloudLink = {
    href: ORIGINAL_CLOUD_LINK,
  };
  const uuids = ['legacy-user-id', 'legacy-session-id'];

  class FixedDate extends Date {
    constructor(...args) {
      super(...(args.length > 0 ? args : [NOW]));
    }

    static now() {
      return NOW;
    }
  }

  class MutationObserver {
    observe() {}
  }

  class TestEvent {
    constructor(type) {
      this.type = type;
    }
  }

  function addEventListener(type, listener) {
    if (!listeners.has(type)) listeners.set(type, []);
    listeners.get(type).push(listener);
  }

  const document = {
    readyState: 'complete',
    body: {},
    addEventListener,
    querySelectorAll() {
      return [cloudLink];
    },
    get cookie() {
      return Array.from(cookies, ([key, value]) => `${key}=${value}`).join('; ');
    },
    set cookie(value) {
      const pair = value.split(';', 1)[0];
      const separator = pair.indexOf('=');
      cookies.set(
        decodeURIComponent(pair.slice(0, separator)),
        decodeURIComponent(pair.slice(separator + 1)),
      );
    },
  };

  const window = {
    location,
    document,
    localStorage: createStorage(),
    sessionStorage: createStorage(),
    crypto: {
      randomUUID() {
        return uuids.shift();
      },
    },
    navigator: {
      userAgent: 'Galaxy compatibility test',
      sendBeacon(url, body) {
        beacons.push({ url, body });
        return true;
      },
    },
    fetch() {
      throw new Error('sendBeacon should handle compatibility-test payloads');
    },
    addEventListener,
    dispatchEvent(event) {
      for (const listener of listeners.get(event.type) || []) listener(event);
    },
    setInterval() {
      return 1;
    },
    clearInterval(id) {
      clearedIntervals.push(id);
    },
    requestAnimationFrame(callback) {
      animationFrames.push(callback);
      return animationFrames.length;
    },
  };
  window.window = window;

  vm.runInNewContext(SCRIPT, {
    Blob,
    Date: FixedDate,
    Event: TestEvent,
    Math,
    MutationObserver,
    Object,
    Promise,
    Uint8Array,
    URL,
    URLSearchParams,
    console,
    document,
    window,
  });

  return {
    beacons,
    clearedIntervals,
    cloudLink,
    dispatch(type, properties = {}) {
      window.dispatchEvent({ type, ...properties });
    },
    runNextAnimationFrame() {
      const callback = animationFrames.shift();
      assert.ok(callback, 'expected a queued animation frame');
      callback();
    },
    window,
  };
}

async function flushRequest(
  environment,
  apiHost = 'https://control-plane-internal.clickhouse.cloud',
  beaconIndex = 0,
) {
  await environment.window.galaxy.flushEvents();
  assert.equal(environment.beacons.length, beaconIndex + 1);
  assert.equal(
    environment.beacons[beaconIndex].url,
    `${apiHost}/api/galaxy?sendGalaxyForensicEvent`,
  );
  return JSON.parse(await environment.beacons[beaconIndex].body.text());
}

test('matches the legacy docs-page request and Cloud-link contract', async () => {
  const environment = createEnvironment(
    '/docs/get-started/setup/install',
    '?utm_source=docs-test&gclid=test-click',
  );

  assert.deepEqual(await flushRequest(environment), LEGACY_CONTRACT.docsPageRequest);
  assert.equal(environment.cloudLink.href, LEGACY_CONTRACT.cloudLink);
});

test('matches the legacy knowledge-base request contract', async () => {
  const environment = createEnvironment(
    '/docs/resources/support-center/knowledge-base/example',
  );

  assert.deepEqual(
    await flushRequest(environment),
    LEGACY_CONTRACT.knowledgeBaseRequest,
  );
});

test('keeps periodic flushing active across a BFCache pagehide', () => {
  const environment = createEnvironment('/docs/get-started/setup/install');

  environment.dispatch('pagehide', { persisted: true });
  assert.deepEqual(environment.clearedIntervals, []);

  environment.dispatch('pagehide', { persisted: false });
  assert.deepEqual(environment.clearedIntervals, [1]);
});

test('tracks a Mintlify SPA route and updates only the current page path', async () => {
  const environment = createEnvironment(
    '/docs/get-started/setup/install',
    '?utm_source=docs-test&gclid=test-click',
  );
  await flushRequest(environment);

  environment.window.location.pathname = '/docs/guides/developer/overview';
  environment.runNextAnimationFrame();

  assert.deepEqual(
    await flushRequest(
      environment,
      'https://control-plane-internal.clickhouse.cloud',
      1,
    ),
    LEGACY_CONTRACT.spaPageRequest,
  );
  assert.equal(environment.cloudLink.href, LEGACY_CONTRACT.spaCloudLink);
});

test('routes preview traffic to the development Galaxy backend', async () => {
  const mintlifyPreview = createEnvironment(
    '/get-started/setup/install',
    '',
    'https://private-docs.mintlify.app',
  );
  await flushRequest(
    mintlifyPreview,
    'https://control-plane-internal.clickhouse-dev.com',
  );
  assert.equal(mintlifyPreview.cloudLink.href, ORIGINAL_CLOUD_LINK);
});

test('keeps Galaxy transport disabled on localhost and unknown origins', async () => {
  const localhost = createEnvironment(
    '/docs/get-started/setup/install',
    '',
    'http://localhost:3000',
  );
  await localhost.window.galaxy.flushEvents();
  assert.deepEqual(localhost.beacons, []);
  assert.equal(localhost.cloudLink.href, ORIGINAL_CLOUD_LINK);

  const unknownOrigin = createEnvironment(
    '/docs/get-started/setup/install',
    '',
    'https://example.com',
  );
  await unknownOrigin.window.galaxy.flushEvents();
  assert.deepEqual(unknownOrigin.beacons, []);
  assert.equal(unknownOrigin.cloudLink.href, ORIGINAL_CLOUD_LINK);
});
