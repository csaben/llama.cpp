"""Build a benchmark-only libllama-common with history proposals and agreement disabled."""
import json
import shlex
import subprocess
from pathlib import Path

repo = Path.cwd()
work = repo / "build/p8m/lookup-off"
work.mkdir(parents=True, exist_ok=True)
source = repo / "common/speculative.cpp"
condition = "dp.prompt && draft_history(dp, model_on, agreement)"
code = source.read_text()
assert code.count(condition) == 1
control = work / source.name
control.write_text(code.replace(condition, "dp.prompt && false"))
entries = json.loads(Path("build/compile_commands.json").read_text())
entry, = (e for e in entries if Path(e["file"]) == source)
obj = str(work / "speculative.cpp.o")
compile_cmd = shlex.split(entry["command"])
compile_cmd[compile_cmd.index(str(source))] = str(control)
compile_cmd[compile_cmd.index("-o") + 1] = obj
subprocess.run(compile_cmd, cwd=entry["directory"], check=True)
link = shlex.split(Path("build/common/CMakeFiles/llama-common.dir/link.txt").read_text())
link[link.index("CMakeFiles/llama-common.dir/speculative.cpp.o")] = obj
link[link.index("-o") + 1] = str(work / "libllama-common.so.1")
subprocess.run(link, cwd=entry["directory"], check=True)
