# Security Policy

## Secret Handling

Never commit real MetaTrader 5 credentials, broker account numbers, passwords,
API tokens, `.env` files, model artifacts, market datasets, logs, reports, or
paper trading databases. Local credential files matching `credentials*.yaml`
are ignored by Git. Use `credentials.example.yaml` as the safe template.

If a real credential was ever placed in a file that may have been shared,
rotate the broker password or revoke the credential before continuing.

## Live Trading Safety

The safe default is paper trading:

```yaml
paper_trading: true
trading_enabled: false
```

Only enable live trading after reviewing the strategy, testing in paper mode,
confirming the MT5 account, and accepting the risk of real capital loss.

## Reporting Issues

For security problems, avoid posting secrets or account details in public
issues. Share only sanitized logs, redacted configs, and reproducible steps.
