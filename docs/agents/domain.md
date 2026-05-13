# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`docs/CONTEXT.md`** at the repo root — 18-term domain glossary for OmniTrade AI
- **`docs/adr/`** — 3 ADRs covering architectural decisions (PaperWallet extraction, EnsembleVoter delegation, unified halt)
- Use the glossary's vocabulary in all output — don't drift to synonyms

## File structure

Single-context repo:

```
/
├── CLAUDE.md
├── docs/
│   ├── CONTEXT.md
│   ├── adr/
│   │   ├── 0001-paper-wallet-extraction.md
│   │   ├── 0002-ensemble-voter-delegation.md
│   │   └── 0003-unified-halt-delegation.md
│   └── agents/
│       ├── issue-tracker.md
│       ├── triage-labels.md
│       └── domain.md
└── omnitrade/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `docs/CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/grill-with-docs`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0001 (PaperWallet extraction) — but worth reopening because…_
