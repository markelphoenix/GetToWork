#!/usr/bin/env python3
"""Code-sign the Windows build's programs and libraries (used by CI once signing is set up).

    python packaging/sign_windows.py dist/GetToWork build/engine [--trusted-signing-dir DIR]

Windows 11's Smart App Control blocks unknown, unsigned programs outright -
even ones Steam installed, with no "Run anyway" (Steam shows error 0x11C7) -
and that includes the bundled ``llama-server.exe`` and its DLLs when the game
starts them. Signed programs also get fewer SmartScreen and antivirus
warnings. So this signs every ``.exe`` and ``.dll`` under the given folders
that isn't validly signed already (Microsoft's own Visual C++ runtime and
``python312.dll`` keep their signatures), with SHA-256 and an RFC 3161
timestamp, using the Windows SDK's ``signtool``.

Two ways to sign, chosen by the environment (never the command line or the
log - GitHub repository secrets):

* **Azure Trusted Signing** (Microsoft's signing service - today's code-signing
  certificates keep their keys in such services or on hardware tokens):
  ``AZURE_TRUSTED_SIGNING_ENDPOINT`` (e.g. https://eus.codesigning.azure.net),
  ``AZURE_TRUSTED_SIGNING_ACCOUNT``, ``AZURE_TRUSTED_SIGNING_PROFILE``, plus the
  service principal's ``AZURE_TENANT_ID``, ``AZURE_CLIENT_ID`` and
  ``AZURE_CLIENT_SECRET`` (read by Microsoft's signing library itself). The
  library (``Azure.CodeSigning.Dlib.dll``, NuGet package
  Microsoft.Trusted.Signing.Client) is looked for in ``--trusted-signing-dir``
  or named by ``AZURE_TRUSTED_SIGNING_DLIB``.
* **A .pfx certificate file** with an exportable key:
  ``WINDOWS_SIGNING_PFX_BASE64`` (the file, base64) and
  ``WINDOWS_SIGNING_PFX_PASSWORD``.

The build workflow runs this only when those secrets exist (never for pull
requests), so builds stay unsigned - and keep working - until signing is set
up. Exit code 0 = signed (or nothing to sign), 1 = failure.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

PFX_ENV = "WINDOWS_SIGNING_PFX_BASE64"
PASSWORD_ENV = "WINDOWS_SIGNING_PFX_PASSWORD"
AZURE_ENDPOINT_ENV = "AZURE_TRUSTED_SIGNING_ENDPOINT"
AZURE_ACCOUNT_ENV = "AZURE_TRUSTED_SIGNING_ACCOUNT"
AZURE_PROFILE_ENV = "AZURE_TRUSTED_SIGNING_PROFILE"
AZURE_DLIB_ENV = "AZURE_TRUSTED_SIGNING_DLIB"
AZURE_CREDENTIAL_ENVS = ("AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET")
DLIB_NAME = "Azure.CodeSigning.Dlib.dll"
TIMESTAMP_URL = "http://timestamp.digicert.com"  # RFC 3161; any trusted timestamp server works
AZURE_TIMESTAMP_URL = "http://timestamp.acs.microsoft.com"
SIGN_SUFFIXES = (".exe", ".dll")
BATCH = 40  # files per signtool call (keeps the command line well under Windows' limit)
_SDK_GLOB = r"C:\Program Files (x86)\Windows Kits\10\bin\*\x64\signtool.exe"


class SignError(Exception):
    """Signing couldn't be done (the message says why - never a password)."""


def find_signtool(which: Callable[[str], Optional[str]] = shutil.which,
                  sdk_glob: str = _SDK_GLOB) -> Optional[str]:
    """signtool.exe: on PATH, else the newest Windows 10/11 SDK's (GitHub's windows runners have it)."""
    found = which("signtool")
    if found:
        return found
    candidates = sorted(glob.glob(sdk_glob), key=lambda p: [int(x) if x.isdigit() else 0
                                                              for x in Path(p).parent.parent.name.split(".")])
    return candidates[-1] if candidates else None


def find_dlib(folder: Optional[Path], env: Mapping[str, str]) -> Optional[Path]:
    """Microsoft's Trusted Signing library: named in the environment, or found in `folder` (x64 preferred)."""
    named = (env.get(AZURE_DLIB_ENV) or "").strip()
    if named:
        return Path(named) if Path(named).is_file() else None
    if folder is None or not folder.is_dir():
        return None
    found = sorted(folder.rglob(DLIB_NAME), key=lambda p: ("x64" not in [part.lower() for part in p.parts], str(p)))
    return found[0] if found else None


def files_to_sign(roots: Sequence[Path]) -> list[Path]:
    """Every .exe and .dll under `roots` (links skipped), in a stable order."""
    found: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix.lower() in SIGN_SUFFIXES:
            found.add(root)
            continue
        for path in root.rglob("*") if root.is_dir() else ():
            if path.suffix.lower() in SIGN_SUFFIXES and path.is_file() and not path.is_symlink():
                found.add(path)
    return sorted(found, key=lambda p: str(p).lower())


def is_signed(signtool: str, path: Path, runner: Callable[..., Any]) -> bool:
    """Does `path` already carry a valid signature (verified against Windows' trusted roots)?"""
    try:
        result = runner([signtool, "verify", "/pa", "/q", str(path)], stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, timeout=120)
    except subprocess.SubprocessError:
        return False
    return getattr(result, "returncode", 1) == 0


def _run_sign(runner: Callable[..., Any], command: list[str], count: int, secret: str) -> None:
    try:
        result = runner(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
    except subprocess.SubprocessError as exc:  # (its message would quote the command line - password and all)
        raise SignError(f"signtool didn't finish signing {count} file(s) ({type(exc).__name__}).") from None
    if getattr(result, "returncode", 1) != 0:
        raw = getattr(result, "stdout", b"") or b""
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        text = text.replace(secret, "***") if secret else text
        raise SignError(f"signtool couldn't sign {count} file(s) (exit code {result.returncode}): "
                        f"{text.strip()[-800:]}")


def signing_setup(env: Mapping[str, str], work: Path, dlib: Optional[Path]) -> tuple[list[str], str, str]:
    """The signtool options for the method the environment sets up: ``(options, timestamp URL, secret)``.

    Azure Trusted Signing when its account is named, else the .pfx file.
    Files written for signtool (the certificate, the service's metadata) go
    into `work`, which the caller deletes.
    """
    account = (env.get(AZURE_ACCOUNT_ENV) or "").strip()
    if account:
        endpoint = (env.get(AZURE_ENDPOINT_ENV) or "").strip()
        profile = (env.get(AZURE_PROFILE_ENV) or "").strip()
        missing = [name for name, value in ((AZURE_ENDPOINT_ENV, endpoint), (AZURE_PROFILE_ENV, profile))
                   if not value] + [name for name in AZURE_CREDENTIAL_ENVS if not (env.get(name) or "").strip()]
        if missing:
            raise SignError(f"Azure Trusted Signing needs {', '.join(missing)} too.")
        if dlib is None:
            raise SignError(f"Microsoft's signing library ({DLIB_NAME}, NuGet package Microsoft.Trusted.Signing."
                            f"Client) wasn't found - pass --trusted-signing-dir or set {AZURE_DLIB_ENV}.")
        metadata = work / "metadata.json"
        metadata.write_text(json.dumps({"Endpoint": endpoint, "CodeSigningAccountName": account,
                                        "CertificateProfileName": profile}), encoding="utf-8")
        return ["/dlib", str(dlib), "/dmdf", str(metadata)], AZURE_TIMESTAMP_URL, ""
    encoded, password = (env.get(PFX_ENV) or "").strip(), env.get(PASSWORD_ENV) or ""
    if not encoded:
        raise SignError(f"Nothing to sign with: set {AZURE_ACCOUNT_ENV} (and the rest of Azure Trusted Signing's "
                        f"settings), or {PFX_ENV} (a .pfx file, base64) and {PASSWORD_ENV}.")
    try:
        certificate = base64.b64decode(encoded, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise SignError(f"{PFX_ENV} isn't valid base64 ({exc}).") from None
    if not certificate:
        raise SignError(f"{PFX_ENV} is empty.")
    pfx = work / "certificate.pfx"
    pfx.write_bytes(certificate)
    return ["/f", str(pfx), "/p", password], TIMESTAMP_URL, password


def sign(roots: Sequence[Path], *, env: Mapping[str, str], runner: Callable[..., Any],
         signtool: Optional[str], dlib: Optional[Path] = None, out: Any = None) -> int:
    """Sign what needs signing under `roots`; returns how many files were signed."""
    out = out if out is not None else sys.stdout
    if not signtool:
        raise SignError("signtool.exe wasn't found (it comes with the Windows SDK).")
    work = Path(tempfile.mkdtemp(prefix="gettowork-sign-"))
    try:
        options, timestamp_url, secret = signing_setup(env, work, dlib)
        files = files_to_sign(roots)
        unsigned = [p for p in files if not is_signed(signtool, p, runner)]
        print(f"{len(files)} programs and libraries found, {len(files) - len(unsigned)} already signed, "
              f"{len(unsigned)} to sign.", file=out, flush=True)
        for start in range(0, len(unsigned), BATCH):
            batch = [str(p) for p in unsigned[start:start + BATCH]]
            _run_sign(runner, [signtool, "sign", "/fd", "SHA256", "/tr", timestamp_url, "/td", "SHA256",
                               *options, *batch], len(batch), secret)
    finally:
        shutil.rmtree(work, ignore_errors=True)  # the certificate never outlives the step
    still = [p for p in unsigned if not is_signed(signtool, p, runner)]
    if still:
        raise SignError("signtool reported success, but these still don't verify: "
                        + ", ".join(p.name for p in still[:5]))
    for path in unsigned:
        print(f"  signed {path}", file=out, flush=True)
    return len(unsigned)


def main(argv: Optional[Sequence[str]] = None, *, env: Optional[Mapping[str, str]] = None,
         runner: Optional[Callable[..., Any]] = None, signtool: Optional[str] = None, out: Any = None) -> int:
    parser = argparse.ArgumentParser(prog="sign_windows.py",
                                     description="Code-sign the Windows build's .exe and .dll files.")
    parser.add_argument("folders", nargs="+", help="folders (or files) to sign, e.g. dist/GetToWork build/engine")
    parser.add_argument("--trusted-signing-dir",
                        help=f"a folder holding Microsoft's {DLIB_NAME} (Azure Trusted Signing)")
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return 0 if exc.code in (0, None) else 1
    environ = os.environ if env is None else env
    try:
        dlib = find_dlib(Path(args.trusted_signing_dir) if args.trusted_signing_dir else None, environ)
        count = sign([Path(f) for f in args.folders], env=environ, runner=runner or subprocess.run,
                     signtool=signtool or find_signtool(), dlib=dlib, out=out)
    except (SignError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"Signed {count} file(s).", file=out if out is not None else sys.stdout, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
