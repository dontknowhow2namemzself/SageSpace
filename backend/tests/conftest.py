"""Test-process defaults that must apply before application imports."""
import os


# Some test modules load ChromaDB/ONNX Runtime during collection. Disable
# third-party telemetry before those imports so tests stay offline and do not
# leave a ``:memory:.ses`` telemetry artifact in the repository.
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
