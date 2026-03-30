# L-S-Equities

Long/Short equity portfolio project developed by QUANTT (Queen's University).

## Quick Start

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Check network/DNS connectivity (required for Yahoo Finance calls):

```bash
python3 scripts/network_doctor.py
```

4. Build universe:

```bash
python3 UniverseBuilder.py
```

## Why DNS Matters Here

`UniverseBuilder.py` uses `yfinance.screen(...)`, which calls Yahoo endpoints. If DNS or outbound HTTPS is blocked, it fails even when your Python code is correct.

You cannot configure DNS servers inside this repository. DNS is controlled by the runtime (your machine, container, or Codespace).

## Codespaces Notes

- This repo includes `.devcontainer/devcontainer.json` so dependencies install automatically.
- Proxy environment variables are passed through if they exist (`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`).
- If `network_doctor.py` fails on Yahoo hosts, check org/network policy and allow outbound HTTPS to:
  - `finance.yahoo.com`
  - `query1.finance.yahoo.com`
  - `guce.yahoo.com`

## Common Commands

```bash
python3 UniverseBuilder.py
python3 StockScreener.py
python3 ls_backtest.py
python3 Analysis.py
```
