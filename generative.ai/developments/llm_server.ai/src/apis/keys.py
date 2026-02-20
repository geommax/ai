"""API key generation utilities."""

import secrets
import string


def generate_api_key(prefix: str = "llm") -> str:
    """Generate a cryptographically secure API key.

    Format: ``llm-<48 random alphanumeric chars>``
    """
    alphabet = string.ascii_letters + string.digits
    body = "".join(secrets.choice(alphabet) for _ in range(48))
    return f"{prefix}-{body}"
