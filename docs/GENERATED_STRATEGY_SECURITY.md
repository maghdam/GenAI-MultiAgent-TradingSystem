# Generated Strategy Trust Boundary

Strategy Studio output is research input, not trusted application code.

## Enforced boundary

- Generated files are not imported during API startup.
- Saving a draft does not register it in the runtime strategy registry.
- Validation parses the Python AST and never executes the source.
- Only the `signals(df, ...)` contract is accepted.
- Only `pandas` and `numpy` imports are accepted.
- Filesystem, network, process, dynamic-code, serialization, and dunder access are rejected by policy.
- Draft and saved backtests execute in a separate isolated Python process.
- The child receives only OHLCV bars and explicit strategy parameters.
- Broker credentials and application environment variables are not passed to the child.
- Execution has a timeout and the result must be a numeric Series aligned to the input bars.

## Important limitation

The subprocess and AST policy are a strong local robustness boundary, but they are not equivalent to an operating-system sandbox or container. Before accepting strategies from untrusted third parties, run the isolated worker in a container with:

- no network
- a read-only filesystem
- CPU and memory limits
- a non-privileged user
- no mounted broker configuration

## Runtime promotion

Generated files cannot directly enter autonomous paper or cTrader execution, regardless of whether the selected broker account is Demo or Live. Lifecycle promotion records evidence and approval state for the exact saved source version, but promotion does not import or register raw generated source into the trusted runtime. Autonomous execution of generated logic still requires explicit reviewed integration into the deterministic runtime strategy registry, tests, and the applicable lifecycle gate. If reviewed runtime logic is later executed against a selected Live account, the normal cTrader safety gates still apply and the resulting orders can use real funds.
