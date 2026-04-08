"""Tool: code_exec — sandboxed Python execution via subprocess."""

import asyncio
import sys
import tempfile
import os


async def code_exec(code: str, timeout: int = 10) -> dict:
    """Execute Python code in a subprocess and return stdout/stderr."""
    timeout = max(1, min(timeout, 30))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return {"error": f"Execution timed out after {timeout}s", "stdout": "", "stderr": ""}

        return {
            "returncode": proc.returncode,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }
    finally:
        os.unlink(tmp_path)
