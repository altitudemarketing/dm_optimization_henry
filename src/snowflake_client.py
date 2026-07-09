"""
Minimal Snowflake SQL API v2 client using JWT key-pair authentication.

Mirrors the auth pattern already in use elsewhere in this project
(speed_snowflake/legacy/validate_bot.py) instead of introducing a new
Snowflake driver dependency. Reads connection details from environment
variables -- see .env.example.

Required environment variables:
  SNOWFLAKE_ACCOUNT_LOCATOR   e.g. BMC55881          (SELECT CURRENT_ACCOUNT();)
  SNOWFLAKE_ACCOUNT_URL       e.g. itagdju-jcc43869   (subdomain of snowflakecomputing.com)
  SNOWFLAKE_USER              service account username
  SNOWFLAKE_KEY_FILE          path to the .p8 private key file for that user
  SNOWFLAKE_ROLE              role to run queries as (needs SELECT on GOOGLE_ADS schema)
  SNOWFLAKE_WAREHOUSE         warehouse to run queries on
"""

import base64
import hashlib
import os
import time
from typing import Any, Dict, List

import jwt
import pandas as pd
import requests
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_private_key,
)

_STATEMENTS_ENDPOINT = "/api/v2/statements"
_POLL_INTERVAL_SECONDS = 1
_MAX_POLL_ATTEMPTS = 60


class SnowflakeConfigError(EnvironmentError):
    """Raised when required Snowflake connection env vars are missing."""


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SnowflakeConfigError(
            f"Missing required environment variable: {name}. See .env.example."
        )
    return value


def _build_jwt(account_locator: str, user: str, key_file: str) -> str:
    with open(key_file, "rb") as f:
        private_key = load_pem_private_key(f.read(), password=None)

    pub_der = private_key.public_key().public_bytes(
        Encoding.DER, PublicFormat.SubjectPublicKeyInfo
    )
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(pub_der).digest()).decode()

    now = int(time.time())
    account = account_locator.upper()
    user_upper = user.upper()
    return jwt.encode(
        {
            "iss": f"{account}.{user_upper}.{fingerprint}",
            "sub": f"{account}.{user_upper}",
            "iat": now,
            "exp": now + 3600,
        },
        private_key,
        algorithm="RS256",
    )


def _base_url(account_url: str) -> str:
    return f"https://{account_url}.snowflakecomputing.com"


def run_query(sql: str, timeout_seconds: int = 120) -> pd.DataFrame:
    """Executes a SQL statement via Snowflake's SQL API v2, returns a DataFrame."""

    account_locator = _require_env("SNOWFLAKE_ACCOUNT_LOCATOR")
    account_url = _require_env("SNOWFLAKE_ACCOUNT_URL")
    user = _require_env("SNOWFLAKE_USER")
    key_file = _require_env("SNOWFLAKE_KEY_FILE")
    role = _require_env("SNOWFLAKE_ROLE")
    warehouse = _require_env("SNOWFLAKE_WAREHOUSE")

    token = _build_jwt(account_locator, user, key_file)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    resp = requests.post(
        _base_url(account_url) + _STATEMENTS_ENDPOINT,
        headers=headers,
        json={"statement": sql, "warehouse": warehouse, "role": role, "timeout": timeout_seconds},
        timeout=timeout_seconds + 10,
    )

    if resp.status_code == 202:
        handle = resp.json()["statementHandle"]
        payload = _poll_until_done(handle, headers, account_url)
    elif resp.status_code == 200:
        payload = resp.json()
    else:
        raise RuntimeError(f"Snowflake query failed ({resp.status_code}): {resp.text}")

    return _payload_to_dataframe(payload, headers, account_url)


def _poll_until_done(handle: str, headers: Dict[str, str], account_url: str) -> Dict[str, Any]:
    url = f"{_base_url(account_url)}{_STATEMENTS_ENDPOINT}/{handle}"
    for _ in range(_MAX_POLL_ATTEMPTS):
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code != 202:
            raise RuntimeError(f"Snowflake polling failed ({resp.status_code}): {resp.text}")
        time.sleep(_POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Snowflake query {handle} did not complete in time.")


def _payload_to_dataframe(
    payload: Dict[str, Any], headers: Dict[str, str], account_url: str
) -> pd.DataFrame:
    columns = [c["name"] for c in payload["resultSetMetaData"]["rowType"]]
    rows: List[List[Any]] = list(payload.get("data", []))

    partition_info = payload["resultSetMetaData"].get("partitionInfo", [])
    handle = payload.get("statementHandle")
    if handle and len(partition_info) > 1:
        for i in range(1, len(partition_info)):
            url = f"{_base_url(account_url)}{_STATEMENTS_ENDPOINT}/{handle}?partition={i}"
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            rows.extend(resp.json().get("data", []))

    return pd.DataFrame(rows, columns=columns)
