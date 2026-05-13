# Extract PaperWallet as shared paper-trading state

TradeExecutor and StockExecutor each duplicated balance tracking, position management, fee/slippage math, and order simulation (~80 lines each). Extracted into a single `PaperWallet` class.

**Why:** The two executors had identical paper-trading logic with different variable names (`_paper_balance` vs `_paper_balance_usd`). When a dust bug was found (sell base_amount recalculated from slippage-adjusted price instead of using exact position amount), it existed in both places. One module = one fix.

**Considered alternative:** Keep duplicated wallet logic per executor. Rejected because: (a) bugs would continue to diverge, (b) adding a third executor would create a third copy, (c) testing balance math requires testing each executor separately.

**Trade-off:** PaperWallet is a new seam — executors depend on it. But the interface is narrow (simulate_order, close_position, get_balance, get_positions) and the implementation is pure math with no external dependencies. The coupling is low-risk.
