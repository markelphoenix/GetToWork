"""HTTPS certificate checks that work on every player's computer.

When the game talks to GitHub (to fetch the llama.cpp engine) or to Jev over
HTTPS, the server proves who it is with a *certificate*, which your computer
checks against a list of trusted authorities (its "trust store").

Python's standard library normally uses the trust store of the OpenSSL
library it was built with. That usually just works - except with the Python
installer from python.org on macOS, which brings its own OpenSSL, ignores the
Mac's Keychain, and has *no* certificates until you run the "Install
Certificates" helper it ships with. Many people never do, and then every
HTTPS request fails with ``CERTIFICATE_VERIFY_FAILED``.

So the game builds its own HTTPS settings:

1. **truststore** (installed alongside ``huggingface_hub``), which asks the
   operating system itself - the macOS Keychain, the Windows certificate store
   or Linux's system bundle (``SSL_CERT_FILE`` is honoured) - just like a web
   browser does;
2. else Python's default trust store *plus* the ``certifi`` bundle;
3. else Python's plain default.

Certificates are always checked: nothing here ever turns verification off.
"""

from __future__ import annotations

import ssl
import urllib.error
from typing import Optional

__all__ = ["https_context", "is_certificate_error", "CERTIFICATE_HELP"]

CERTIFICATE_HELP = (
    "This isn't a network problem: your Python couldn't check the server's security certificate. "
    "On a Mac with Python from python.org, open your Python folder in Applications and double-click "
    "'Install Certificates.command', then try again. On a work or school network that inspects secure "
    "connections, ask IT for its certificate and point the SSL_CERT_FILE environment variable at it."
)

_CONTEXT: Optional[ssl.SSLContext] = None


def https_context(*, fresh: bool = False) -> ssl.SSLContext:
    """An SSLContext that verifies certificates against a trust store that works here.

    Built once and reused (pass ``fresh=True`` for a new one, e.g. in tests).
    """
    global _CONTEXT
    if _CONTEXT is not None and not fresh:
        return _CONTEXT
    context: Optional[ssl.SSLContext] = None
    try:
        import truststore  # the operating system's own trust store

        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        context = None
    if context is None:
        context = ssl.create_default_context()
        try:
            import certifi  # Mozilla's list of trusted authorities, as a file

            context.load_verify_locations(cafile=certifi.where())
        except Exception:
            pass  # the plain default is still better than nothing
    _CONTEXT = context
    return context


def is_certificate_error(exc: BaseException) -> bool:
    """True if `exc` (or the error inside a urllib ``URLError``) is a failed certificate check."""
    seen = 0
    current: Optional[BaseException] = exc
    while current is not None and seen < 5:
        if isinstance(current, ssl.SSLCertVerificationError):
            return True
        if isinstance(current, ssl.SSLError) and "CERTIFICATE_VERIFY_FAILED" in str(current):
            return True
        reason = getattr(current, "reason", None) if isinstance(current, urllib.error.URLError) else None
        if isinstance(reason, BaseException):
            current = reason
        else:
            current = current.__cause__ or current.__context__
        seen += 1
    return False
