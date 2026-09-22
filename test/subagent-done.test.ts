import { describe, it, afterEach } from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { createServer } from "node:http";
import { join } from "node:path";
import { tmpdir } from "node:os";

import {
  parseDeniedTools,
  shouldAutoExitOnAgentEnd,
  shouldMarkUserTookOver,
  writeExitSidecar,
} from "../subagent-done.ts";
import { writeContextUsageSidecar } from "../src/context-usage.ts";
import { writeDeciderMode } from "../src/idle-decider.ts";
import {
  clearActiveSubagents,
  markSubagentActive,
} from "../src/runtime-state.ts";

const cleanups: Array<() => void> = [];
afterEach(() => {
  clearActiveSubagents();
  while (cleanups.length > 0) cleanups.pop()!();
});

describe("subagent-done: shouldMarkUserTookOver", () => {
  it("ignores the initial injected task before the first agent run", () => {
    assert.equal(shouldMarkUserTookOver(false), false);
  });

  it("treats later input as manual takeover", () => {
    assert.equal(shouldMarkUserTookOver(true), true);
  });
});

describe("subagent-done: shouldAutoExitOnAgentEnd", () => {
  it("auto-exits after normal completion when there was no takeover", () => {
    const messages = [{ role: "assistant", stopReason: "stop" }];
    assert.equal(shouldAutoExitOnAgentEnd(false, messages), true);
  });

  it("auto-exits after normal completion even when the user sent the prompt", () => {
    const messages = [{ role: "assistant", stopReason: "stop" }];
    assert.equal(shouldAutoExitOnAgentEnd(true, messages), true);
  });

  it("stays open after Escape aborts the run", () => {
    const messages = [{ role: "assistant", stopReason: "aborted" }];
    assert.equal(shouldAutoExitOnAgentEnd(false, messages), false);
  });

  it("stays open on API errors (timeout, connection failure) so pi can retry", () => {
    const messages = [{ role: "assistant", stopReason: "error", errorMessage: "Request timed out." }];
    assert.equal(shouldAutoExitOnAgentEnd(false, messages), false);
  });

  it("stays open on connection errors so pi can retry", () => {
    const messages = [{ role: "assistant", stopReason: "error", errorMessage: "Connection error: WebSocket error" }];
    assert.equal(shouldAutoExitOnAgentEnd(false, messages), false);
  });

  it("defaults to exiting when no messages are available", () => {
    assert.equal(shouldAutoExitOnAgentEnd(false, undefined), true);
  });

  it("stays open when the turn produced no new assistant message (errored/retrying turn)", () => {
    // Resumed-session failure mode (verified live, pi 0.80.3): the resume
    // message is delivered, the first request times out, pi schedules a retry,
    // and agent_end fires with the conversation ending at the just-delivered
    // USER message. Walking backwards would find the PREVIOUS conversation's
    // assistant (stopReason "stop") and shut pi down mid-retry.
    const messages = [
      { role: "assistant", stopReason: "stop" }, // stale: pre-resume history
      { role: "user" }, // the resume message — no reply yet
    ];
    assert.equal(shouldAutoExitOnAgentEnd(false, messages), false);
  });

  it("still auto-exits when a completed turn follows a resumed conversation", () => {
    const messages = [
      { role: "assistant", stopReason: "stop" },
      { role: "user" },
      { role: "assistant", stopReason: "toolUse" },
      { role: "toolResult" },
      { role: "assistant", stopReason: "stop" },
    ];
    assert.equal(shouldAutoExitOnAgentEnd(false, messages), true);
  });

  it("stays open while nested subagents are still running", () => {
    const messages = [{ role: "assistant", stopReason: "stop" }];
    assert.equal(shouldAutoExitOnAgentEnd(false, messages, 2), false);
  });
});

describe("subagent-done: parseDeniedTools", () => {
  it("splits and trims comma-separated names, dropping empties", () => {
    assert.deepEqual(parseDeniedTools(" subagent , subagent_resume ,,bash "), [
      "subagent",
      "subagent_resume",
      "bash",
    ]);
  });

  it("returns an empty list when unset", () => {
    assert.deepEqual(parseDeniedTools(undefined), []);
  });
});

describe("subagent-done: .exit sidecar shapes (cross-extension contract)", () => {
  function makeSessionFile(): string {
    const dir = mkdtempSync(join(tmpdir(), "herdr-done-"));
    cleanups.push(() => rmSync(dir, { recursive: true, force: true }));
    return join(dir, "child.jsonl");
  }

  it('done writes exactly {"type":"done"}', () => {
    const sessionFile = makeSessionFile();
    writeExitSidecar(sessionFile, { type: "done" });
    assert.equal(readFileSync(`${sessionFile}.exit`, "utf8"), '{"type":"done"}');
  });

  it("ping writes type/name/message in reference byte order", () => {
    const sessionFile = makeSessionFile();
    writeExitSidecar(sessionFile, { type: "ping", name: "Worker", message: "need input" });
    assert.equal(
      readFileSync(`${sessionFile}.exit`, "utf8"),
      '{"type":"ping","name":"Worker","message":"need input"}',
    );
  });

  it("publishes context usage atomically with version and subagent id", () => {
    const sessionFile = makeSessionFile();
    assert.equal(
      writeContextUsageSidecar(sessionFile, "child-1", {
        tokens: 75_000,
        contextWindow: 200_000,
        percent: 37.5,
      }),
      true,
    );
    assert.deepEqual(JSON.parse(readFileSync(`${sessionFile}.context-usage`, "utf8")), {
      version: 1,
      subagentId: "child-1",
      tokens: 75_000,
      contextWindow: 200_000,
      percent: 37.5,
    });
    assert.deepEqual(
      readdirSync(join(sessionFile, "..")),
      ["child.jsonl.context-usage"],
      "the temporary file is renamed away",
    );
  });
});

describe("subagent-done: module", () => {
  it("loads standalone and exports a default extension factory", async () => {
    const mod = await import("../subagent-done.ts");
    assert.equal(typeof mod.default, "function");
  });
});

describe("subagent-done: subagent_done tool writes sidecar and shuts down", () => {
  function makeSessionFile(): string {
    const dir = mkdtempSync(join(tmpdir(), "herdr-done-"));
    cleanups.push(() => rmSync(dir, { recursive: true, force: true }));
    return join(dir, "child.jsonl");
  }

  it("does not recreate a consumed sidecar when agent_end follows subagent_done", async () => {
    const sessionFile = makeSessionFile();
    const origSession = process.env.PI_SUBAGENT_SESSION;
    const origAutoExit = process.env.PI_SUBAGENT_AUTO_EXIT;
    process.env.PI_SUBAGENT_SESSION = sessionFile;
    process.env.PI_SUBAGENT_AUTO_EXIT = "1";
    cleanups.push(() => {
      if (origSession !== undefined) process.env.PI_SUBAGENT_SESSION = origSession;
      else delete process.env.PI_SUBAGENT_SESSION;
      if (origAutoExit !== undefined) process.env.PI_SUBAGENT_AUTO_EXIT = origAutoExit;
      else delete process.env.PI_SUBAGENT_AUTO_EXIT;
    });

    const handlers: Record<string, Function> = {};
    const registeredTools: Record<string, any> = {};
    const fakePi = {
      on: (event: string, handler: Function) => { handlers[event] = handler; },
      registerTool: (tool: any) => { registeredTools[tool.name] = tool; },
      registerShortcut: () => {},
      getAllTools: () => [],
    };
    const fakeCtx = {
      shutdown: () => {},
      getContextUsage: () => undefined,
      ui: { setWidget: () => {} },
    };

    const mod = await import("../subagent-done.ts");
    mod.default(fakePi as any);
    await registeredTools.subagent_done.execute("call-1", {}, null, () => {}, fakeCtx);
    assert.equal(existsSync(`${sessionFile}.exit`), true, "tool writes the terminal sidecar");

    rmSync(`${sessionFile}.exit`);
    handlers.agent_end?.(
      { messages: [{ role: "assistant", stopReason: "stop" }] },
      fakeCtx,
    );
    assert.equal(
      existsSync(`${sessionFile}.exit`),
      false,
      "agent_end must not recreate a sidecar already consumed by the watcher",
    );
  });

  it("writes usage before the exact done sidecar and shuts down", async () => {
    const sessionFile = makeSessionFile();
    const origSession = process.env.PI_SUBAGENT_SESSION;
    const origId = process.env.PI_SUBAGENT_ID;
    process.env.PI_SUBAGENT_SESSION = sessionFile;
    process.env.PI_SUBAGENT_ID = "child-1";
    cleanups.push(() => {
      if (origSession !== undefined) process.env.PI_SUBAGENT_SESSION = origSession;
      else delete process.env.PI_SUBAGENT_SESSION;
      if (origId !== undefined) process.env.PI_SUBAGENT_ID = origId;
      else delete process.env.PI_SUBAGENT_ID;
    });

    const registeredTools: Record<string, any> = {};
    let shutdownCalled = false;
    const fakePi = {
      on: () => {},
      registerTool: (tool: any) => { registeredTools[tool.name] = tool; },
      registerShortcut: () => {},
      getAllTools: () => [],
    };
    const fakeCtx = {
      shutdown: () => {
        assert.ok(existsSync(`${sessionFile}.context-usage`), "usage is published before shutdown");
        assert.ok(existsSync(`${sessionFile}.exit`), "terminal signal is published before shutdown");
        shutdownCalled = true;
      },
      getContextUsage: () => ({ tokens: 75_000, contextWindow: 200_000, percent: 37.5 }),
      ui: { setWidget: () => {} },
    };

    const mod = await import("../subagent-done.ts");
    mod.default(fakePi as any);

    assert.ok(registeredTools.subagent_done, "subagent_done tool should be registered");
    await registeredTools.subagent_done.execute("call-1", {}, null, () => {}, fakeCtx);

    assert.equal(shutdownCalled, true, "should have called shutdown");
    const sidecar = readFileSync(`${sessionFile}.exit`, "utf8");
    assert.equal(sidecar, '{"type":"done"}', "should write done sidecar");
    assert.deepEqual(JSON.parse(readFileSync(`${sessionFile}.context-usage`, "utf8")), {
      version: 1,
      subagentId: "child-1",
      tokens: 75_000,
      contextWindow: 200_000,
      percent: 37.5,
    });
  });
});

describe("subagent-done: user close without subagent_done leaves no sidecar", () => {
  function makeSessionFile(): string {
    const dir = mkdtempSync(join(tmpdir(), "herdr-done-"));
    cleanups.push(() => rmSync(dir, { recursive: true, force: true }));
    return join(dir, "child.jsonl");
  }

  it("no sidecar written when agent_end fires after abort (user Escape)", async () => {
    const sessionFile = makeSessionFile();
    const origSession = process.env.PI_SUBAGENT_SESSION;
    const origAutoExit = process.env.PI_SUBAGENT_AUTO_EXIT;
    process.env.PI_SUBAGENT_SESSION = sessionFile;
    process.env.PI_SUBAGENT_AUTO_EXIT = "1";
    cleanups.push(() => {
      if (origSession !== undefined) process.env.PI_SUBAGENT_SESSION = origSession;
      else delete process.env.PI_SUBAGENT_SESSION;
      if (origAutoExit !== undefined) process.env.PI_SUBAGENT_AUTO_EXIT = origAutoExit;
      else delete process.env.PI_SUBAGENT_AUTO_EXIT;
    });

    const handlers: Record<string, Function> = {};
    let shutdownCalled = false;
    const fakePi = {
      on: (event: string, handler: Function) => { handlers[event] = handler; },
      registerTool: () => {},
      registerShortcut: () => {},
      getAllTools: () => [],
    };
    const fakeCtx = {
      shutdown: () => { shutdownCalled = true; },
      ui: { setWidget: () => {} },
    };

    const mod = await import("../subagent-done.ts");
    mod.default(fakePi as any);

    handlers.session_start?.({}, fakeCtx);
    handlers.agent_start?.();
    // User aborts — should NOT auto-exit, no sidecar
    handlers.agent_end?.(
      { messages: [{ role: "assistant", stopReason: "aborted" }] },
      fakeCtx,
    );

    assert.equal(shutdownCalled, false, "should NOT shutdown on abort");
    let sidecarExists = false;
    try { readFileSync(`${sessionFile}.exit`); sidecarExists = true; } catch {}
    assert.equal(sidecarExists, false, "should NOT write sidecar on user abort");
  });
});

describe("subagent-done: session_shutdown context usage fallback", () => {
  function makeSessionFile(): string {
    const dir = mkdtempSync(join(tmpdir(), "herdr-done-"));
    cleanups.push(() => rmSync(dir, { recursive: true, force: true }));
    return join(dir, "child.jsonl");
  }

  it("writes once on user shutdown and does not overwrite the first valid snapshot", async () => {
    const sessionFile = makeSessionFile();
    const origSession = process.env.PI_SUBAGENT_SESSION;
    const origId = process.env.PI_SUBAGENT_ID;
    process.env.PI_SUBAGENT_SESSION = sessionFile;
    process.env.PI_SUBAGENT_ID = "child-shutdown";
    cleanups.push(() => {
      if (origSession === undefined) delete process.env.PI_SUBAGENT_SESSION;
      else process.env.PI_SUBAGENT_SESSION = origSession;
      if (origId === undefined) delete process.env.PI_SUBAGENT_ID;
      else process.env.PI_SUBAGENT_ID = origId;
    });

    const handlers: Record<string, Function> = {};
    const fakePi = {
      on: (event: string, handler: Function) => { handlers[event] = handler; },
      registerTool: () => {},
      registerShortcut: () => {},
      getAllTools: () => [],
    };
    let usage = { tokens: 10, contextWindow: 100, percent: 10 };
    const fakeCtx = {
      getContextUsage: () => usage,
      ui: { setWidget: () => {} },
    };

    const mod = await import("../subagent-done.ts");
    mod.default(fakePi as any);
    handlers.session_shutdown?.({}, fakeCtx);
    usage = { tokens: 90, contextWindow: 100, percent: 90 };
    handlers.session_shutdown?.({}, fakeCtx);

    assert.deepEqual(JSON.parse(readFileSync(`${sessionFile}.context-usage`, "utf8")), {
      version: 1,
      subagentId: "child-shutdown",
      tokens: 10,
      contextWindow: 100,
      percent: 10,
    });
    assert.equal(existsSync(`${sessionFile}.exit`), false, "fallback does not alter terminal signals");
  });

  it("skips unavailable usage without throwing", async () => {
    const sessionFile = makeSessionFile();
    const origSession = process.env.PI_SUBAGENT_SESSION;
    const origId = process.env.PI_SUBAGENT_ID;
    process.env.PI_SUBAGENT_SESSION = sessionFile;
    process.env.PI_SUBAGENT_ID = "child-unknown";
    cleanups.push(() => {
      if (origSession === undefined) delete process.env.PI_SUBAGENT_SESSION;
      else process.env.PI_SUBAGENT_SESSION = origSession;
      if (origId === undefined) delete process.env.PI_SUBAGENT_ID;
      else process.env.PI_SUBAGENT_ID = origId;
    });

    const handlers: Record<string, Function> = {};
    const fakePi = {
      on: (event: string, handler: Function) => { handlers[event] = handler; },
      registerTool: () => {},
      registerShortcut: () => {},
      getAllTools: () => [],
    };
    const mod = await import("../subagent-done.ts");
    mod.default(fakePi as any);

    assert.doesNotThrow(() => handlers.session_shutdown?.({}, { getContextUsage: () => undefined }));
    assert.equal(existsSync(`${sessionFile}.context-usage`), false);
  });
});

describe("subagent-done: agent_end writes .exit sidecar on clean auto-exit", () => {
  function makeSessionFile(): string {
    const dir = mkdtempSync(join(tmpdir(), "herdr-done-"));
    cleanups.push(() => rmSync(dir, { recursive: true, force: true }));
    return join(dir, "child.jsonl");
  }

  it("writes done sidecar when agent_end triggers auto-exit", async () => {
    const sessionFile = makeSessionFile();
    const origSession = process.env.PI_SUBAGENT_SESSION;
    const origAutoExit = process.env.PI_SUBAGENT_AUTO_EXIT;
    process.env.PI_SUBAGENT_SESSION = sessionFile;
    process.env.PI_SUBAGENT_AUTO_EXIT = "1";
    cleanups.push(() => {
      if (origSession !== undefined) process.env.PI_SUBAGENT_SESSION = origSession;
      else delete process.env.PI_SUBAGENT_SESSION;
      if (origAutoExit !== undefined) process.env.PI_SUBAGENT_AUTO_EXIT = origAutoExit;
      else delete process.env.PI_SUBAGENT_AUTO_EXIT;
    });

    const handlers: Record<string, Function> = {};
    let shutdownCalled = false;
    const fakePi = {
      on: (event: string, handler: Function) => { handlers[event] = handler; },
      registerTool: () => {},
      registerShortcut: () => {},
      getAllTools: () => [],
    };
    const fakeCtx = {
      shutdown: () => { shutdownCalled = true; },
      ui: { setWidget: () => {} },
    };

    const mod = await import("../subagent-done.ts");
    mod.default(fakePi as any);

    // Simulate session_start to initialize
    handlers.session_start?.({}, fakeCtx);
    // Simulate agent_start so agentStarted = true
    handlers.agent_start?.();
    // Simulate agent_end with a clean completion
    handlers.agent_end?.(
      { messages: [{ role: "assistant", stopReason: "stop" }] },
      fakeCtx,
    );

    assert.equal(shutdownCalled, true, "should have called shutdown");
    const sidecar = readFileSync(`${sessionFile}.exit`, "utf8");
    assert.equal(sidecar, '{"type":"done"}', "should write done sidecar on auto-exit");
  });

  it("does NOT write done sidecar when agent_end is an error (retry)", async () => {
    const sessionFile = makeSessionFile();
    const origSession = process.env.PI_SUBAGENT_SESSION;
    const origAutoExit = process.env.PI_SUBAGENT_AUTO_EXIT;
    process.env.PI_SUBAGENT_SESSION = sessionFile;
    process.env.PI_SUBAGENT_AUTO_EXIT = "1";
    cleanups.push(() => {
      if (origSession !== undefined) process.env.PI_SUBAGENT_SESSION = origSession;
      else delete process.env.PI_SUBAGENT_SESSION;
      if (origAutoExit !== undefined) process.env.PI_SUBAGENT_AUTO_EXIT = origAutoExit;
      else delete process.env.PI_SUBAGENT_AUTO_EXIT;
    });

    const handlers: Record<string, Function> = {};
    let shutdownCalled = false;
    const fakePi = {
      on: (event: string, handler: Function) => { handlers[event] = handler; },
      registerTool: () => {},
      registerShortcut: () => {},
      getAllTools: () => [],
    };
    const fakeCtx = {
      shutdown: () => { shutdownCalled = true; },
      ui: { setWidget: () => {} },
    };

    const mod = await import("../subagent-done.ts");
    mod.default(fakePi as any);

    handlers.session_start?.({}, fakeCtx);
    handlers.agent_start?.();
    handlers.agent_end?.(
      { messages: [{ role: "assistant", stopReason: "error", errorMessage: "Request timed out." }] },
      fakeCtx,
    );

    assert.equal(shutdownCalled, false, "should NOT shutdown on error");
    let sidecarExists = false;
    try { readFileSync(`${sessionFile}.exit`); sidecarExists = true; } catch {}
    assert.equal(sidecarExists, false, "should NOT write sidecar on error");
  });

  it("does NOT auto-exit an orchestrator while nested subagents are running", async () => {
    const sessionFile = makeSessionFile();
    const origSession = process.env.PI_SUBAGENT_SESSION;
    const origAutoExit = process.env.PI_SUBAGENT_AUTO_EXIT;
    process.env.PI_SUBAGENT_SESSION = sessionFile;
    process.env.PI_SUBAGENT_AUTO_EXIT = "1";
    markSubagentActive("nested-1");
    markSubagentActive("nested-2");
    cleanups.push(() => {
      if (origSession !== undefined) process.env.PI_SUBAGENT_SESSION = origSession;
      else delete process.env.PI_SUBAGENT_SESSION;
      if (origAutoExit !== undefined) process.env.PI_SUBAGENT_AUTO_EXIT = origAutoExit;
      else delete process.env.PI_SUBAGENT_AUTO_EXIT;
    });

    const handlers: Record<string, Function> = {};
    let shutdownCalled = false;
    const fakePi = {
      on: (event: string, handler: Function) => { handlers[event] = handler; },
      registerTool: () => {},
      registerShortcut: () => {},
      getAllTools: () => [],
    };
    const fakeCtx = {
      shutdown: () => { shutdownCalled = true; },
      ui: { setWidget: () => {} },
    };

    const mod = await import("../subagent-done.ts");
    mod.default(fakePi as any);
    handlers.agent_end?.(
      { messages: [{ role: "assistant", stopReason: "stop" }] },
      fakeCtx,
    );

    assert.equal(shutdownCalled, false, "should keep the nested orchestrator alive");
    let sidecarExists = false;
    try { readFileSync(`${sessionFile}.exit`); sidecarExists = true; } catch {}
    assert.equal(sidecarExists, false, "must not signal completion before children settle");
  });
});

describe("subagent-done: Laya decides exit vs keep-open", () => {
  const reply = [{ role: "assistant", stopReason: "stop", content: [{ type: "text", text: "report" }] }];

  // Fake laya/server.py: /decide answers `done`, /label requests are recorded.
  async function fakeLaya(done: boolean) {
    const labels: any[] = [];
    const server = createServer((req, res) => {
      let body = "";
      req.on("data", (c) => (body += c));
      req.on("end", () => {
        if (req.url === "/label") labels.push(JSON.parse(body));
        res.end(JSON.stringify({ done }));
      });
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    cleanups.push(() => server.close());
    return { url: `http://127.0.0.1:${(server.address() as any).port}`, labels };
  }

  async function launch(env: Record<string, string>) {
    const dir = mkdtempSync(join(tmpdir(), "herdr-done-"));
    const sessionFile = join(dir, "child.jsonl");
    // Isolated HOME (decider mode + key files) and no real TypeSafe key from the dev shell.
    const all = {
      PI_SUBAGENT_SESSION: sessionFile,
      PI_SUBAGENT_AUTO_EXIT: "0",
      HOME: dir,
      TYPESAFE_API_KEY: "",
      JEV_API_KEY: "",
      ...env,
    };
    const orig = Object.fromEntries(Object.keys(all).map((k) => [k, process.env[k]]));
    Object.assign(process.env, all);
    cleanups.push(() => {
      rmSync(dir, { recursive: true, force: true });
      for (const [k, v] of Object.entries(orig)) v === undefined ? delete process.env[k] : (process.env[k] = v);
    });

    const handlers: Record<string, Function> = {};
    const state = { shutdown: false };
    const ctx = { shutdown: () => { state.shutdown = true; }, ui: { setWidget: () => {} } };
    const mod = await import("../subagent-done.ts");
    mod.default({
      on: (e: string, h: Function) => { handlers[e] = h; },
      registerTool: () => {},
      registerShortcut: () => {},
      getAllTools: () => [],
    } as any);
    handlers.agent_start({}, ctx);
    return { handlers, ctx, state, sessionFile };
  }

  it("closes an interactive subagent Laya judges done", async () => {
    const laya = await fakeLaya(true);
    const { handlers, ctx, state, sessionFile } = await launch({ PI_LAYA_URL: laya.url });
    await handlers.agent_end({ messages: reply }, ctx);
    assert.equal(state.shutdown, true);
    assert.equal(readFileSync(`${sessionFile}.exit`, "utf8"), '{"type":"done"}');
  });

  it("keeps an auto-exit subagent open when Laya sees a question, labels it on reply", async () => {
    const laya = await fakeLaya(false);
    const { handlers, ctx, state, sessionFile } = await launch({
      PI_LAYA_URL: laya.url,
      PI_SUBAGENT_AUTO_EXIT: "1",
    });
    await handlers.agent_end({ messages: reply }, ctx);
    assert.equal(state.shutdown, false);
    assert.equal(existsSync(`${sessionFile}.exit`), false);

    handlers.input({}, ctx);
    for (let i = 0; i < 50 && laya.labels.length === 0; i++) await new Promise((r) => setTimeout(r, 10));
    assert.deepEqual(laya.labels, [{ text: "report", done: false }]);
  });

  it("labels done when a kept-open pane closes without input", async () => {
    const laya = await fakeLaya(false);
    const { handlers, ctx } = await launch({ PI_LAYA_URL: laya.url });
    await handlers.agent_end({ messages: reply }, ctx);
    await handlers.session_shutdown({}, { ...ctx, getContextUsage: () => undefined });
    assert.deepEqual(laya.labels, [{ text: "report", done: true }]);
  });

  it("falls back to the autoExit flag when the sidecar is unreachable", async () => {
    const { handlers, ctx, state } = await launch({
      PI_LAYA_URL: "http://127.0.0.1:1",
      PI_SUBAGENT_AUTO_EXIT: "1",
    });
    await handlers.agent_end({ messages: reply }, ctx);
    assert.equal(state.shutdown, true);
  });

  it("never shuts down a turn the user started while Laya was unreachable", async () => {
    const { handlers, ctx, state } = await launch({
      PI_LAYA_URL: "http://127.0.0.1:1",
      PI_SUBAGENT_AUTO_EXIT: "1",
    });
    const pending = handlers.agent_end({ messages: reply }, ctx);
    handlers.input({}, ctx); // typed during the await
    await pending;
    assert.equal(state.shutdown, false);
  });

  it("gives up on a kept-open auto-exit subagent after the idle timeout, unlabelled", async () => {
    const laya = await fakeLaya(false);
    const { handlers, ctx, state, sessionFile } = await launch({
      PI_LAYA_URL: laya.url,
      PI_SUBAGENT_AUTO_EXIT: "1",
      PI_LAYA_IDLE_SECS: "0.05",
    });
    await handlers.agent_end({ messages: reply }, ctx);
    assert.equal(state.shutdown, false);
    await new Promise((r) => setTimeout(r, 120));
    assert.equal(state.shutdown, true);
    assert.equal(readFileSync(`${sessionFile}.exit`, "utf8"), '{"type":"done"}');
    await handlers.session_shutdown({}, { ...ctx, getContextUsage: () => undefined });
    assert.deepEqual(laya.labels, []);
  });

  it("input cancels the idle timeout", async () => {
    const laya = await fakeLaya(false);
    const { handlers, ctx, state } = await launch({
      PI_LAYA_URL: laya.url,
      PI_SUBAGENT_AUTO_EXIT: "1",
      PI_LAYA_IDLE_SECS: "0.05",
    });
    await handlers.agent_end({ messages: reply }, ctx);
    handlers.input({}, ctx);
    await new Promise((r) => setTimeout(r, 120));
    assert.equal(state.shutdown, false);
  });

  it("mode off consults no classifier", async () => {
    const laya = await fakeLaya(true);
    const { handlers, ctx, state } = await launch({ PI_LAYA_URL: laya.url });
    writeDeciderMode("off");
    await handlers.agent_end({ messages: reply }, ctx);
    assert.equal(state.shutdown, false, "interactive + no classifier stays open");
  });

  // Fake api.typesafe.ai: answers p(finished), records request bodies.
  async function fakeJev(finished: number, status = 200) {
    const bodies: any[] = [];
    const server = createServer((req, res) => {
      let body = "";
      req.on("data", (c) => (body += c));
      req.on("end", () => {
        bodies.push(JSON.parse(body));
        res.statusCode = status;
        res.end(JSON.stringify({ answers: { q: { probabilities: { finished } } } }));
      });
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    cleanups.push(() => server.close());
    return { url: `http://127.0.0.1:${(server.address() as any).port}`, bodies };
  }

  it("jev mode sends brief + reply + last turn and acts on the answer", async () => {
    const jev = await fakeJev(0.9);
    const { handlers, ctx, state } = await launch({ PI_JEV_URL: jev.url, TYPESAFE_API_KEY: "k" });
    // Resumed session: this run's event only has the resume prompt; the brief is
    // in the session, as it is in the replayed training data.
    const session = [
      { role: "user", content: [{ type: "text", text: "Write the report. " }] },
      { role: "assistant", stopReason: "stop", content: [{ type: "text", text: "draft" }] },
      { role: "user", content: [{ type: "text", text: "continue" }] },
      { role: "assistant", stopReason: "toolUse", content: [] },
      { role: "toolResult", toolName: "bash", isError: true },
      ...reply,
    ];
    const sessionManager = {
      getEntries: () => [{ type: "session" }, ...session.map((message) => ({ type: "message", message }))],
    };
    await handlers.agent_end({ messages: session.slice(2) }, { ...ctx, sessionManager });
    assert.equal(state.shutdown, true);
    assert.deepEqual(jev.bodies[0].state, {
      final_message: "report",
      last_turn: "The final turn ran 1 tool calls and 1 failed. The last tool call (bash) failed.",
      task: "Write the report.",
    });
  });

  it("falls back to Laya when Jev errors", async () => {
    const jev = await fakeJev(0.9, 500);
    const laya = await fakeLaya(false);
    const { handlers, ctx, state } = await launch({
      PI_JEV_URL: jev.url,
      TYPESAFE_API_KEY: "k",
      PI_LAYA_URL: laya.url,
      PI_SUBAGENT_AUTO_EXIT: "1",
    });
    await handlers.agent_end({ messages: reply }, ctx);
    assert.equal(jev.bodies.length, 1);
    assert.equal(state.shutdown, false, "Laya's keep-open wins over the autoExit flag");
    handlers.input({}, ctx); // clears the idle timer
  });

  it("skips Laya on interactive turns the user started", async () => {
    const laya = await fakeLaya(true);
    const { handlers, ctx, state } = await launch({ PI_LAYA_URL: laya.url });
    handlers.input({}, ctx); // after agent_start → user took over
    await handlers.agent_end({ messages: reply }, ctx);
    assert.equal(state.shutdown, false);
  });
});
