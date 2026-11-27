// Regression: the client once stored the native fetch on `this` and called it as a method,
// which browsers reject with "Illegal invocation" -- so every request failed before leaving
// the page, and the deployed UI always said "API unreachable". Typechecking and building
// could not see it. This fetch stand-in throws the same way when called as a method.
import assert from "node:assert/strict";
import test from "node:test";

import { RepurposeMapClient } from "../src/client.ts";

test("the default fetch is called detached, not as a method of the client", async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = function strictFetch() {
    if (this !== undefined && this !== globalThis) throw new TypeError("Illegal invocation");
    return Promise.resolve(new Response(JSON.stringify({ status: "ok" }), { status: 200 }));
  };
  try {
    const client = new RepurposeMapClient({ baseUrl: "http://example.invalid/" });
    assert.deepEqual(await client.health(), { status: "ok" });
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("an injected fetchFn is used as given, and the base URL has no double slash", async () => {
  let seen;
  const client = new RepurposeMapClient({
    baseUrl: "http://example.invalid/",
    fetchFn: async (url) => {
      seen = url;
      return new Response("{}", { status: 200 });
    },
  });
  await client.health();
  assert.equal(seen, "http://example.invalid/health");
});
