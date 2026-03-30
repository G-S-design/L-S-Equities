#!/usr/bin/env python3
"""
Quick DNS/HTTPS diagnostics for this project.

Use this before running UniverseBuilder.py in a new environment.
"""

from __future__ import annotations

import os
import socket
import sys
import urllib.request
from typing import Iterable


REQUIRED_HOSTS = [
    "finance.yahoo.com",
    "query1.finance.yahoo.com",
    "guce.yahoo.com",
]

GENERAL_HOSTS = [
    "pypi.org",
    "github.com",
]


def check_dns(host: str) -> tuple[bool, str]:
    try:
        # If this resolves, DNS is functioning for this host.
        socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        return True, "resolved"
    except Exception as exc:  # broad on purpose for CLI diagnostics
        return False, f"dns failed: {exc}"


def check_https(url: str, timeout: int = 6) -> tuple[bool, str]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, "status", 200)
            return True, f"http {code}"
    except Exception as exc:  # broad on purpose for CLI diagnostics
        return False, f"https failed: {exc}"


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def run_host_checks(hosts: Iterable[str]) -> tuple[int, int]:
    total = 0
    failed = 0
    for host in hosts:
        total += 1
        dns_ok, dns_msg = check_dns(host)
        https_ok, https_msg = check_https(f"https://{host}")
        ok = dns_ok and https_ok
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"{status:4}  {host:28}  {dns_msg}; {https_msg}")
    return total, failed


def main() -> int:
    print("Network Doctor")
    print("==============")

    print_section("Proxy Environment Variables")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
        value = os.environ.get(name) or os.environ.get(name.lower())
        redacted = "<set>" if value else "<not set>"
        print(f"{name:12} {redacted}")

    print_section("Required Yahoo Hosts")
    req_total, req_failed = run_host_checks(REQUIRED_HOSTS)

    print_section("General Connectivity Hosts")
    gen_total, gen_failed = run_host_checks(GENERAL_HOSTS)

    print_section("Summary")
    print(f"Required hosts: {req_total - req_failed}/{req_total} passed")
    print(f"General hosts:  {gen_total - gen_failed}/{gen_total} passed")

    if req_failed > 0:
        print(
            "\nAction: DNS/outbound network is blocking Yahoo hosts. "
            "This cannot be fixed in repo code alone."
        )
        print(
            "If you are in Codespaces, check org/network policy or set proxy "
            "env vars (HTTP_PROXY/HTTPS_PROXY/NO_PROXY)."
        )
        return 2

    print("\nAll required Yahoo checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
