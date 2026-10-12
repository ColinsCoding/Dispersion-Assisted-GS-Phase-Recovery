"""List running Python processes on Windows (flagging Jupyter kernels) and
stop one by PID.

Leaked notebook kernels hold on to memory after a failed nbconvert run;
enough of them exhaust the Windows commit limit and new kernels die on
start-up ("kernel died", WinError 1455). This script shows what is
running so the idle ones can be stopped deliberately, one PID at a time.

Usage:
    py -3.13 scripts/kernels.py list
    py -3.13 scripts/kernels.py kill 12345
(or: make kernels / make kill-pid PID=12345)
"""
import json
import os
import subprocess
import sys

QUERY = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
         "Select-Object ProcessId,Name,WorkingSetSize,CommandLine | ConvertTo-Json -Compress")


def list_python_processes():
    """Return one dict per running python*.exe process, largest memory first."""
    result = subprocess.run(["powershell", "-NoProfile", "-Command", QUERY],
                            capture_output=True, text=True, check=True)
    text = result.stdout.strip()
    if not text:
        return []
    data = json.loads(text)
    if isinstance(data, dict):          # PowerShell returns a bare object when there is only one
        data = [data]
    processes = []
    for item in data:
        command = item.get("CommandLine") or ""
        pid = int(item["ProcessId"])
        if pid in (os.getpid(), os.getppid()):
            kind = "this script"
        elif "ipykernel" in command:
            kind = "jupyter kernel"
        else:
            kind = "python"
        processes.append({"pid": pid, "name": item["Name"], "kind": kind, "command": command,
                          "mb": int(item.get("WorkingSetSize") or 0) / 2**20})
    return sorted(processes, key=lambda process: process["mb"], reverse=True)


def print_table(processes):
    """Print PID, memory, kind and a shortened command line for each process."""
    if not processes:
        print("no python processes are running")
        return
    print(f"{'PID':>7}  {'MB':>7}  {'kind':14}  command")
    for process in processes:
        print(f"{process['pid']:>7}  {process['mb']:>7.0f}  {process['kind']:14}  {process['command'][:90]}")
    kernels = [process for process in processes if process["kind"] == "jupyter kernel"]
    print(f"\n{len(kernels)} jupyter kernel(s). A kernel using little memory and not attached to an open")
    print("notebook is probably left over. Stop one with:  make kill-pid PID=<pid>")


def kill(pid_text):
    """Stop one python process by PID. Refuses anything that is not a python process."""
    try:
        pid = int(pid_text)
    except ValueError:
        print(f"PID must be a whole number, got {pid_text!r}")
        return 2
    matches = [process for process in list_python_processes() if process["pid"] == pid]
    if not matches:
        print(f"PID {pid} is not a running python process; nothing was stopped (run 'list' to see PIDs)")
        return 2
    if matches[0]["kind"] == "this script":
        print("refusing to stop this script's own process")
        return 2
    print(f"stopping PID {pid} ({matches[0]['kind']}): {matches[0]['command'][:90]}")
    result = subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, text=True)
    print((result.stdout or result.stderr).strip())
    return result.returncode


def main(argv):
    if len(argv) == 1 and argv[0] == "list":
        print_table(list_python_processes())
        return 0
    if len(argv) == 2 and argv[0] == "kill":
        return kill(argv[1])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
