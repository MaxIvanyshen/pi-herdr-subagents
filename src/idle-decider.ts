/**
 * Idle decision shared by the orchestrator's /subagent-decider command and the
 * child extension (subagent-done.ts): should a subagent whose turn just ended
 * exit as done, or stay open for the user?
 *
 * Modes (persisted in ~/.pi/laya/mode, read by children on every decision):
 *   jev  — TypeSafe Jev, given the brief + final reply + last-turn tool summary;
 *          falls back to the local Laya sidecar when there's no key or Jev fails.
 *   laya — local Laya sidecar only (laya/server.py), final reply only. Nothing leaves the machine.
 *   off  — no classifier; the agent's auto-exit flag decides, as before.
 *
 * Numbers behind the choices are in laya/compare.py (Jev +task+turn: AUC 0.993 on
 * replayed sessions + seeds).
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export type DeciderMode = "jev" | "laya" | "off";
export const DECIDER_MODES: DeciderMode[] = ["jev", "laya", "off"];

const JEV_URL = "https://api.typesafe.ai/v1/systemone";
// Jev's p(finished) separates cleanly around here; fit in compare.py with a
// wrong keep-open costing 5x a wrong exit.
export const JEV_THRESHOLD = 0.065;

// Lazily resolved so tests can point HOME elsewhere.
const layaDir = () => join(homedir(), ".pi", "laya");

export function readDeciderMode(): DeciderMode {
  try {
    const mode = readFileSync(join(layaDir(), "mode"), "utf8").trim();
    if ((DECIDER_MODES as string[]).includes(mode)) return mode as DeciderMode;
  } catch {}
  return "jev";
}

export function writeDeciderMode(mode: DeciderMode): void {
  mkdirSync(layaDir(), { recursive: true });
  writeFileSync(join(layaDir(), "mode"), mode + "\n");
}

/**
 * Children don't get the orchestrator's full env, so besides the env vars a
 * key can live in ~/.pi/laya/typesafe-key (chmod 600).
 */
export const jevKeyFile = () => join(layaDir(), "typesafe-key");

export function readJevKey(): string | undefined {
  const fromEnv = process.env.TYPESAFE_API_KEY || process.env.JEV_API_KEY;
  if (fromEnv) return fromEnv;
  try {
    return readFileSync(jevKeyFile(), "utf8").trim() || undefined;
  } catch {
    return undefined;
  }
}

// Mirrors msg_text() in laya/server.py: string content as-is, part lists joined and trimmed.
function messageText(msg: any): string {
  if (typeof msg?.content === "string") return msg.content;
  return (msg?.content ?? [])
    .filter((c: any) => c?.type === "text")
    .map((c: any) => c.text)
    .join("\n")
    .trim();
}

/** The brief: text of the first user message in the session. */
export function firstUserText(messages: any[] | undefined): string {
  const first = messages?.find((m) => m?.role === "user");
  return first ? messageText(first) : "";
}

/**
 * Tool outcomes of the latest turn, in words — Jev/Laya can't compare numbers.
 * Must match last_turn() in laya/server.py, which produced the training data.
 */
export function lastTurnSummary(messages: any[] | undefined): string {
  const results: any[] = [];
  for (let i = (messages?.length ?? 0) - 1; i >= 0; i--) {
    const msg = messages![i];
    if (msg?.role === "user") break;
    if (msg?.role === "toolResult") results.unshift(msg);
  }
  if (results.length === 0) return "The final turn used no tools.";
  const failed = results.filter((r) => r.isError).length;
  const last = results[results.length - 1];
  return (
    `The final turn ran ${results.length} tool calls and ${failed || "none"} failed. ` +
    `The last tool call (${last.toolName}) ${last.isError ? "failed" : "succeeded"}.`
  );
}

/** Same question compare.py measured as "+task+turn". */
const JEV_QUESTION = {
  type: "choice",
  instructions:
    "The author was given the task below. What state is their work in at the end of their final message?",
  criteria: {
    finished: "the task as given is done, including tasks that only asked for a report, review or options",
    blocked: "the work stopped, waiting for the reader's answer, decision, approval or missing information",
    unsure: "the result may be wrong or incomplete and the reader should check it",
  },
};

/**
 * Ask Jev whether the subagent is finished. null = no answer (network, auth,
 * timeout) — the caller falls back.
 */
export async function jevDecide(
  key: string,
  input: { finalMessage: string; lastTurn: string; task: string },
  url = JEV_URL,
): Promise<boolean | null> {
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { authorization: `Bearer ${key}`, "content-type": "application/json" },
      body: JSON.stringify({
        model: "jev-latest",
        // Same field order and truncation as compare.py.
        state: {
          final_message: input.finalMessage.slice(-1500),
          last_turn: input.lastTurn,
          task: input.task.slice(0, 1000),
        },
        questions: { q: JEV_QUESTION },
      }),
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) return null;
    const p = (await res.json())?.answers?.q?.probabilities?.finished;
    return typeof p === "number" ? p >= JEV_THRESHOLD : null;
  } catch {
    return null;
  }
}
