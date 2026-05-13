# Unified halt: RiskManager delegates halt state to SafetyGuard

RiskManager and SafetyGuard both had independent `_trading_halted` flags. When SafetyGuard triggered a kill switch (consecutive losses, equity drawdown), RiskManager didn't know about it — `validate_trade()` would approve trades that SafetyGuard would then block at the gate. Refactored so RiskManager's `_trading_halted` is a Python `@property` that delegates to `SafetyGuard.is_halted` when a SafetyGuard is wired.

**Why:** Two sources of truth for "should we be trading?" created a race: RiskManager could say yes while SafetyGuard said no. The system needs exactly one halt authority. SafetyGuard is the narrower gate (per-trade checks) and already owns the kill switch, so it's the natural authority.

**Trade-off:** RiskManager now has a soft dependency on SafetyGuard (via an optional `safety` parameter). When `safety=None`, RiskManager uses its own halt flags (standalone mode, backward-compatible). When wired, it delegates. The `@property` indirection means reading `_trading_halted` has a conditional branch — negligible cost, but worth noting.

**Considered alternative:** Have the caller check both flags (`if rm._halted or sg.is_halted`). Rejected because it pushes the responsibility to every call site — exactly the kind of scattered logic that causes drift.
