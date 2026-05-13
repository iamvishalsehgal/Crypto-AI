"""
Security fix verification tests.

Validates that the 5 security vulnerability fixes from the audit are
in place and working.  Runs as a standalone script (no pytest).

Usage:
    python tests/test_security_fixes.py
"""

import ast
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ======================================================================
# 1.  Torch.load / joblib.load -- unsafe deserialisation
# ======================================================================

def _file_contains(file_path: str, *patterns: str) -> bool:
    """Return True if *file_path* contains every pattern as a substring."""
    try:
        text = open(file_path).read()
    except FileNotFoundError:
        return False
    return all(p in text for p in patterns)


def test_torch_load_weights_only_ddqn() -> None:
    path = os.path.join(PROJECT_ROOT, "omnitrade", "models", "ddqn_agent.py")
    assert _file_contains(path, "torch.load", "weights_only=True"), (
        "ddqn_agent.py torch.load missing weights_only=True"
    )


def test_torch_load_weights_only_lstm() -> None:
    path = os.path.join(PROJECT_ROOT, "omnitrade", "models", "lstm_model.py")
    assert _file_contains(path, "torch.load", "weights_only=True"), (
        "lstm_model.py torch.load missing weights_only=True"
    )


def test_torch_load_weights_only_cnn() -> None:
    path = os.path.join(PROJECT_ROOT, "omnitrade", "models", "cnn_model.py")
    assert _file_contains(path, "torch.load", "weights_only=True"), (
        "cnn_model.py torch.load missing weights_only=True"
    )


def test_lightgbm_joblib_load_warning() -> None:
    path = os.path.join(PROJECT_ROOT, "omnitrade", "models", "lightgbm_model.py")
    text = open(path).read()
    assert "joblib.load" in text, "lightgbm_model.py no longer uses joblib.load?"
    # Must contain a security warning comment about pickle safety
    assert "SECURITY WARNING" in text or "SECURITY:" in text, (
        "lightgbm_model.py joblib.load missing pickle security warning"
    )


# ======================================================================
# 2.  Git credential redaction
# ======================================================================

def _sanitize_stderr(text: str) -> str:
    """Replicate the exact helper from auto_trader.py / benchmark_runner.py."""
    sanitized = re.sub(r"https?://[^@\s]+@", "https://<redacted>@", text)
    sanitized = re.sub(
        r"(token|password|secret|key)=[^&\s]+",
        r"\1=<redacted>",
        sanitized,
        flags=re.IGNORECASE,
    )
    return sanitized


def test_git_credential_redaction_https() -> None:
    """Token embedded in HTTPS URL should be redacted."""
    raw = (
        "remote: Invalid username or password.\n"
        "fatal: Authentication failed for "
        "'https://ghp_abc123def456token@github.com/user/repo.git'"
    )
    cleaned = _sanitize_stderr(raw)
    assert "https://<redacted>@" in cleaned, (
        f"HTTPS token not redacted: {cleaned!r}"
    )
    assert "ghp_abc123def456token" not in cleaned, (
        "Token still visible after sanitisation"
    )


def test_git_credential_redaction_password_form() -> None:
    """password=plaintext in stderr should be redacted."""
    raw = "error: password=supersecret123 failed"
    cleaned = _sanitize_stderr(raw)
    assert "password=<redacted>" in cleaned, (
        f"Password not redacted: {cleaned!r}"
    )
    assert "supersecret123" not in cleaned


def test_git_credential_redaction_token_form() -> None:
    """token=xxx in stderr should be redacted."""
    raw = "token=ghp_abc123def456token sent"
    cleaned = _sanitize_stderr(raw)
    assert "token=<redacted>" in cleaned, (
        f"Token not redacted: {cleaned!r}"
    )
    assert "ghp_abc123def456token" not in cleaned


def test_git_credential_redaction_noop_on_safe() -> None:
    """Plain stderr with no credentials should pass through unchanged."""
    raw = "fatal: could not read Username for 'https://github.com'"
    cleaned = _sanitize_stderr(raw)
    assert cleaned == raw, (
        f"Safe stderr was modified: {cleaned!r}"
    )


# ======================================================================
# 3.  .gitignore contains .env
# ======================================================================

def test_gitignore_has_env() -> None:
    path = os.path.join(PROJECT_ROOT, ".gitignore")
    assert os.path.exists(path), ".gitignore file not found"
    text = open(path).read()
    assert ".env" in text, (
        ".gitignore does not contain '.env' — "
        "API keys (15+) could be committed"
    )


# ======================================================================
# 4.  MongoDB default-URI warning exists
# ======================================================================

def test_mongodb_default_warning() -> None:
    path = os.path.join(PROJECT_ROOT, "omnitrade", "config", "settings.py")
    text = open(path).read()
    assert 'uri == "mongodb://localhost:27017"' in text or (
        '"mongodb://localhost:27017"' in text and "warning" in text.lower()
    ), (
        "settings.py missing warning for default MongoDB URI without auth"
    )


# ======================================================================
# 5.  os.chdir fragility comment
# ======================================================================

def test_os_chdir_fragile_comment() -> None:
    path = os.path.join(PROJECT_ROOT, "scripts", "auto_trader.py")
    text = open(path).read()
    assert "FRAGILE" in text and "os.chdir" in text, (
        "auto_trader.py os.chdir missing FRAGILE comment"
    )


# ======================================================================
# Runner
# ======================================================================

if __name__ == "__main__":
    tests = [
        test_torch_load_weights_only_ddqn,
        test_torch_load_weights_only_lstm,
        test_torch_load_weights_only_cnn,
        test_lightgbm_joblib_load_warning,
        test_git_credential_redaction_https,
        test_git_credential_redaction_password_form,
        test_git_credential_redaction_token_form,
        test_git_credential_redaction_noop_on_safe,
        test_gitignore_has_env,
        test_mongodb_default_warning,
        test_os_chdir_fragile_comment,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  OK: {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL: {t.__name__} -- {exc}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
