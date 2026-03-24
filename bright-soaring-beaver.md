# Plan: Fix Worker Using Wrong Model (Settings Not Persisted)

## Context

Workers are using `anthropic/claude-sonnet-4` (OpenRouter's hardcoded default) instead of the user's chosen model (`moonshotai/kimi-k2`).

**Root cause:** The UI at `bench-tab.tsx:447` displays the model as `settings.llm?.managerModel || BASELINE_MANAGER_MODEL` where `BASELINE_MANAGER_MODEL = 'moonshotai/kimi-k2'`. This means the dropdown *shows* Kimi K2 even when no model has been saved to the DB. Since the user never explicitly changed the dropdown, the model was never persisted. On the server side, `getModelFromSettings()` returns `undefined`, and the OpenRouter provider falls back to its default `anthropic/claude-sonnet-4`.

## Changes

### 1. Add server-side default model fallback

**File:** `apps/api/src/lib/agents/worker/base.ts`

In `getModelFromSettings()` (line 168-170), add a fallback to a default model when settings don't specify one:

```typescript
protected override getModelFromSettings(ctx: AgentContext): string | undefined {
  return ctx.settings?.llm?.workerModel ?? ctx.settings?.llm?.managerModel ?? DEFAULT_WORKER_MODEL;
}
```

**File:** `apps/api/src/lib/agents/manager.ts`

Same for manager (line 298-300):

```typescript
protected override getModelFromSettings(ctx: AgentContext): string | undefined {
  return ctx.settings?.llm?.managerModel ?? DEFAULT_MANAGER_MODEL;
}
```

**File:** `apps/api/src/lib/llm/models.ts`

Export the default model constant (single source of truth):

```typescript
export const DEFAULT_MODEL = 'moonshotai/kimi-k2';
```

### 2. MiniMax model (already done)

The MiniMax model was already added to `KNOWN_MODELS` in a previous change.

## Files to modify

- `apps/api/src/lib/llm/models.ts` — add `DEFAULT_MODEL` export
- `apps/api/src/lib/agents/worker/base.ts` — import and use `DEFAULT_MODEL` as fallback
- `apps/api/src/lib/agents/manager.ts` — import and use `DEFAULT_MODEL` as fallback
- `apps/web/src/components/apps/bench-tab.tsx` — use the same constant or keep `BASELINE_MANAGER_MODEL` in sync

## Verification

1. `bun run typecheck` passes
2. Without any settings saved, worker logs should show `Using model from project settings: moonshotai/kimi-k2`
3. Changing the dropdown to a different model → worker uses the new model
