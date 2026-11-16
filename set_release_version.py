"""Sets each model's Core ML version to the release tag. stage_huggingface_release.sh runs it on its copies in
huggingface_release/, so every release carries its tag:

    python set_release_version.py v1.2 huggingface_release/*.mlpackage

swift-moonshine-coreml reads the version at load (modelDescription.metadata[.versionString]) and fails unless all five
models carry one release it was tested with. The version is stored in the model, so it stays with it when Xcode compiles
it into an app. Only the package's model spec is rewritten, with metadata.versionString set: the weights and every other
file stay byte for byte as they were.

The spec is written with its map entries in sorted order (deterministic=True). Otherwise protobuf writes them in an
order that changes from run to run, so the same release staged twice would differ byte for byte. v1.1 was stamped
before this script existed, so its spec files hold the same messages in another order.
"""

import json
import sys
from pathlib import Path

from coremltools.proto import Model_pb2


def model_spec_path(package_path: Path) -> Path:
    """The .mlmodel file inside a .mlpackage, as its Manifest.json names it."""
    manifest = json.loads((package_path / "Manifest.json").read_text())
    root_model_item = manifest["itemInfoEntries"][manifest["rootModelIdentifier"]]
    return package_path / "Data" / root_model_item["path"]


def set_release_version(package_path: Path, release_tag: str) -> None:
    spec_path = model_spec_path(package_path)
    spec = Model_pb2.Model()
    spec.ParseFromString(spec_path.read_bytes())
    spec.description.metadata.versionString = release_tag
    spec_path.write_bytes(spec.SerializeToString(deterministic=True))

    written_spec = Model_pb2.Model()
    written_spec.ParseFromString(spec_path.read_bytes())
    if written_spec.description.metadata.versionString != release_tag:
        raise RuntimeError(f"{package_path}: the version reads back as {written_spec.description.metadata.versionString!r}")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: python set_release_version.py <release tag, such as v1.2> <package.mlpackage>...")
    release_tag, *package_paths = sys.argv[1:]
    for package_path in map(Path, package_paths):
        set_release_version(package_path, release_tag)
        print(f"  {package_path.name}: version {release_tag}")


if __name__ == "__main__":
    main()
