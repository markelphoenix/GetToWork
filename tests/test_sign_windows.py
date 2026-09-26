"""Tests for packaging/sign_windows.py, which code-signs the Windows build once a certificate is set up.

No signtool here: a fake runner answers ``signtool verify`` / ``signtool sign``.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sign_windows", ROOT / "packaging" / "sign_windows.py")
sw = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(sw)

PASSWORD = "hunter2-very-secret"
PFX = b"\x30\x82 fake pkcs12 bytes"
ENV = {sw.PFX_ENV: base64.b64encode(PFX).decode(), sw.PASSWORD_ENV: PASSWORD}


class FakeSigntool:
    """`verify` succeeds for files already signed (or signed since); `sign` marks its files signed."""

    def __init__(self, signed=(), fail_sign=False, still_unsigned=()):
        self.signed = {Path(p).name for p in signed}
        self.fail_sign = fail_sign
        self.still_unsigned = set(still_unsigned)
        self.calls: list[list[str]] = []
        self.pfx_seen: list[bytes] = []

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        if args[1] == "verify":
            return subprocess.CompletedProcess(args, 0 if Path(args[-1]).name in self.signed else 1)
        assert args[1] == "sign"
        pfx = Path(args[args.index("/f") + 1])
        self.pfx_seen.append(pfx.read_bytes())
        if self.fail_sign:
            return subprocess.CompletedProcess(args, 1, stdout=f"SignTool Error: bad password {PASSWORD}".encode())
        files = args[args.index("SHA256", args.index("/td")) + 1:]
        files = files[files.index("/p") + 2:]
        self.signed |= {Path(f).name for f in files if Path(f).name not in self.still_unsigned}
        return subprocess.CompletedProcess(args, 0, stdout=b"Done Adding Additional Store")


def windows_build(tmp_path: Path) -> list[Path]:
    game = tmp_path / "dist" / "GetToWork"
    engine = tmp_path / "build" / "engine" / "b11100-vulkan"
    files = [game / "GetToWork.exe", game / "gettowork-cli.exe", game / "_internal" / "python312.dll",
             game / "_internal" / "tcl86t.dll", engine / "llama-server.exe", engine / "ggml-vulkan.dll",
             engine / "vcruntime140.dll"]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"MZ")
    (engine / "install.json").write_text("{}")
    (game / "_internal" / "base_library.zip").write_bytes(b"PK")
    return files


def run(tmp_path, runner, env=ENV):
    out = io.StringIO()
    code = sw.main([str(tmp_path / "dist" / "GetToWork"), str(tmp_path / "build" / "engine")], env=env,
                   runner=runner, signtool="signtool.exe", out=out)
    return code, out.getvalue()


def test_every_unsigned_program_and_library_is_signed_and_signed_ones_are_kept(tmp_path, capsys):
    """Smart App Control checks every .exe and .dll as it loads - llama-server.exe and the ggml DLLs included.
    Files Microsoft (or the PSF) already signed keep their own signature."""
    files = windows_build(tmp_path)
    tool = FakeSigntool(signed=("python312.dll", "vcruntime140.dll"))
    code, out = run(tmp_path, tool)
    assert code == 0, out + capsys.readouterr().err
    sign_calls = [c for c in tool.calls if c[1] == "sign"]
    assert len(sign_calls) == 1
    signed = {Path(f).name for f in sign_calls[0][sign_calls[0].index("/p") + 2:]}
    assert signed == {"GetToWork.exe", "gettowork-cli.exe", "tcl86t.dll", "llama-server.exe", "ggml-vulkan.dll"}
    call = sign_calls[0]
    assert call[call.index("/fd") + 1] == "SHA256" and call[call.index("/td") + 1] == "SHA256"
    assert call[call.index("/tr") + 1] == sw.TIMESTAMP_URL
    assert tool.pfx_seen == [PFX]
    assert not Path(call[call.index("/f") + 1]).exists()  # the certificate never outlives the step
    assert PASSWORD not in out and PASSWORD not in capsys.readouterr().err
    assert "7 programs and libraries found, 2 already signed, 5 to sign." in out
    assert len(files) == 7


AZURE_ENV = {sw.AZURE_ENDPOINT_ENV: "https://eus.codesigning.azure.net", sw.AZURE_ACCOUNT_ENV: "gettowork",
             sw.AZURE_PROFILE_ENV: "public-trust", "AZURE_TENANT_ID": "t", "AZURE_CLIENT_ID": "c",
             "AZURE_CLIENT_SECRET": PASSWORD}


class FakeAzureSigntool(FakeSigntool):
    def __call__(self, args, **kwargs):
        if args[1] == "sign":
            self.calls.append(list(args))
            metadata = Path(args[args.index("/dmdf") + 1])
            self.metadata = json.loads(metadata.read_text())
            self.signed |= {Path(f).name for f in args[args.index("/dmdf") + 2:]}
            return subprocess.CompletedProcess(args, 0, stdout=b"ok")
        return super().__call__(args, **kwargs)


def test_azure_trusted_signing_uses_microsofts_library_and_the_service_metadata(tmp_path, capsys):
    """Today's code-signing keys live in services like Azure Trusted Signing (no exportable .pfx): signtool
    signs through Microsoft's library, told the endpoint/account/profile in a metadata file."""
    windows_build(tmp_path)
    dlib = tmp_path / "trusted-signing" / "Microsoft.Trusted.Signing.Client.1.0.86" / "bin" / "x64" / sw.DLIB_NAME
    dlib.parent.mkdir(parents=True)
    dlib.write_bytes(b"MZ")
    (dlib.parent.parent / "x86").mkdir()
    (dlib.parent.parent / "x86" / sw.DLIB_NAME).write_bytes(b"MZ")
    tool = FakeAzureSigntool(signed=("python312.dll", "vcruntime140.dll"))
    out = io.StringIO()
    code = sw.main([str(tmp_path / "dist" / "GetToWork"), str(tmp_path / "build" / "engine"),
                    "--trusted-signing-dir", str(tmp_path / "trusted-signing")],
                   env=AZURE_ENV, runner=tool, signtool="signtool.exe", out=out)
    assert code == 0, capsys.readouterr().err
    call = next(c for c in tool.calls if c[1] == "sign")
    assert Path(call[call.index("/dlib") + 1]) == dlib  # the x64 library
    assert call[call.index("/tr") + 1] == sw.AZURE_TIMESTAMP_URL and "/p" not in call and "/f" not in call
    assert tool.metadata == {"Endpoint": "https://eus.codesigning.azure.net", "CodeSigningAccountName": "gettowork",
                             "CertificateProfileName": "public-trust"}
    assert not Path(call[call.index("/dmdf") + 1]).exists()
    assert PASSWORD not in " ".join(call) and PASSWORD not in out.getvalue()


def test_azure_trusted_signing_says_what_is_missing(tmp_path, capsys):
    windows_build(tmp_path)
    partial = {k: v for k, v in AZURE_ENV.items() if k not in (sw.AZURE_PROFILE_ENV, "AZURE_CLIENT_SECRET")}
    assert run(tmp_path, FakeAzureSigntool(), env=partial)[0] == 1
    err = capsys.readouterr().err
    assert sw.AZURE_PROFILE_ENV in err and "AZURE_CLIENT_SECRET" in err
    assert run(tmp_path, FakeAzureSigntool(), env=AZURE_ENV)[0] == 1  # (no library found)
    assert sw.DLIB_NAME in capsys.readouterr().err


def test_nothing_to_sign_is_fine(tmp_path):
    windows_build(tmp_path)
    tool = FakeSigntool(signed=[p.name for p in windows_build(tmp_path)])
    code, out = run(tmp_path, tool)
    assert code == 0 and not [c for c in tool.calls if c[1] == "sign"]
    assert "0 to sign" in out


def test_a_signing_failure_fails_the_build_without_showing_the_password(tmp_path, capsys):
    windows_build(tmp_path)
    code, _out = run(tmp_path, FakeSigntool(fail_sign=True))
    err = capsys.readouterr().err
    assert code == 1 and "signtool couldn't sign" in err
    assert PASSWORD not in err and "***" in err


def test_a_file_that_still_doesnt_verify_fails(tmp_path, capsys):
    windows_build(tmp_path)
    code, _out = run(tmp_path, FakeSigntool(still_unsigned=("llama-server.exe",)))
    assert code == 1 and "llama-server.exe" in capsys.readouterr().err


def test_a_timeout_never_prints_the_command_line(tmp_path, capsys):
    windows_build(tmp_path)

    def slow(args, **kwargs):
        if args[1] == "verify":
            return subprocess.CompletedProcess(args, 1)
        raise subprocess.TimeoutExpired(args, 1800)

    code, _out = run(tmp_path, slow)
    err = capsys.readouterr().err
    assert code == 1 and "TimeoutExpired" in err and PASSWORD not in err


@pytest.mark.parametrize("env, words", [
    ({}, sw.AZURE_ACCOUNT_ENV),
    ({sw.PFX_ENV: "", sw.PASSWORD_ENV: "x"}, sw.PFX_ENV),
    ({sw.PFX_ENV: "!!!not base64!!!", sw.PASSWORD_ENV: "x"}, "isn't valid base64"),
])
def test_without_a_usable_certificate_it_says_what_to_set(tmp_path, capsys, env, words):
    windows_build(tmp_path)
    code, _out = run(tmp_path, FakeSigntool(), env=env)
    assert code == 1 and words in capsys.readouterr().err


def test_finding_signtool(tmp_path):
    assert sw.find_signtool(which=lambda name: r"C:\tools\signtool.exe") == r"C:\tools\signtool.exe"
    for version in ("10.0.19041.0", "10.0.22621.0", "10.0.9200.0"):
        tool = tmp_path / "Kits" / "bin" / version / "x64" / "signtool.exe"
        tool.parent.mkdir(parents=True)
        tool.write_bytes(b"MZ")
    found = sw.find_signtool(which=lambda name: None, sdk_glob=str(tmp_path / "Kits" / "bin" / "*" / "x64" /
                                                                   "signtool.exe"))
    assert found is not None and "10.0.22621.0" in found
    assert sw.find_signtool(which=lambda name: None, sdk_glob=str(tmp_path / "none" / "*.exe")) is None


def test_the_build_workflow_signs_only_when_a_certificate_is_set_up():
    yaml = pytest.importorskip("yaml")

    data = yaml.safe_load((ROOT / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8"))
    job = data["jobs"]["standalone"]
    assert job["env"]["HAS_WINDOWS_SIGNING"] == ("${{ secrets.AZURE_TRUSTED_SIGNING_ACCOUNT != '' || "
                                                 "secrets.WINDOWS_SIGNING_PFX_BASE64 != '' }}")
    names = [s.get("name", "") for s in job["steps"]]
    step = next(s for s in job["steps"] if s.get("name", "").startswith("Code-sign the Windows programs"))
    assert "runner.os == 'Windows'" in step["if"] and "env.HAS_WINDOWS_SIGNING == 'true'" in step["if"]
    assert "github.event_name != 'pull_request'" in step["if"]
    names_needed = (sw.PFX_ENV, sw.PASSWORD_ENV, sw.AZURE_ENDPOINT_ENV, sw.AZURE_ACCOUNT_ENV, sw.AZURE_PROFILE_ENV,
                    *sw.AZURE_CREDENTIAL_ENVS)
    assert step["env"] == {name: "${{ secrets.%s }}" % name for name in names_needed}
    assert "packaging/sign_windows.py dist/GetToWork build/engine" in step["run"]
    assert "nuget install Microsoft.Trusted.Signing.Client" in step["run"] and "--trusted-signing-dir" in step["run"]
    # After PyInstaller and the engine fetch, before the archive is packed (and tested).
    assert names.index("Build the game with PyInstaller") < names.index(step["name"])
    assert names.index(step["name"]) < next(i for i, n in enumerate(names) if n.startswith("Add the engine"))
