"""Run the upstream noncausal regression using its local tiny GGUF fixture."""
import os
import sys
from pathlib import Path

root = Path.cwd()
sys.path[:0] = [str(root / "tools/server/tests"), str(root / "tools/server/tests/unit")]
import test_systemone

model = Path(sys.argv[1]).resolve(strict=True)
os.environ.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="1", PORT="18435",
                  LLAMA_SERVER_BIN_PATH=str(root / "build/bin/llama-server"))
preset = test_systemone.ServerPreset.tinydecision


def local_model(name):
    server = preset(name)
    server.model_hf_repo = None
    server.model_file = str(model)
    return server


test_systemone.ServerPreset.tinydecision = local_model
try:
    test_systemone.test_systemone_noncausal_parallel()
finally:
    for server in list(test_systemone.server_instances):
        server.stop()
print("Serial and four parallel noncausal requests agreed within 1e-6.")
