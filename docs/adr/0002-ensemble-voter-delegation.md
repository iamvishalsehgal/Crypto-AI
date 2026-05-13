# Delegate StockModelFactory voting to EnsembleVoter via adapters

StockModelFactory had ~40 lines of custom majority-voting logic that duplicated EnsembleVoter's weighted aggregation. Refactored to create an internal `EnsembleVoter`, register each model with an Adapter, and delegate `predict()` to `voter.vote()`.

**Why:** Two voting implementations existed — one in EnsembleVoter (crypto lane) and one in StockModelFactory (stock lane). They used different tie-breaking rules, different confidence calculations, and different signal normalisation. EnsembleVoter already handled weighted voting, signal normalisation, and confidence aggregation correctly.

**Trade-off:** Each model needs a thin Adapter class (~15 lines) to translate its output format to the voter's expected BUY/SELL/HOLD string. This is a bit more code than calling `model.predict()` directly, but the adapters are trivial and the voting logic is guaranteed consistent across lanes.

**Considered alternative:** Keep the two voting systems separate and manually sync changes. Rejected because EnsembleVoter was already being maintained as the canonical voter — every improvement to it would need to be mirrored in StockModelFactory's custom logic, which had already drifted.
