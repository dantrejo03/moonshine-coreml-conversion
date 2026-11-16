"""The plan's Conversion rules, shared by every step's convert.py: fixed-shape inputs (float32, or int32 for integers;
Step 5's float16 package also takes float16), float32 outputs, mlprogram, watchOS 11, and "convert twice" (float32 on
CPU only to prove the math, float16 on all compute units for the watch)."""

import collections
from pathlib import Path

import coremltools
import numpy as np
from coremltools.models.compute_plan import MLComputePlan

from common.config import CONVERSION_DIRECTORY


def convert_to_coreml(
    traced_wrapper,
    input_shapes: dict[str, tuple[int, ...]],
    output_names: list[str],
    compute_precision,
    compute_units,
    input_data_types: dict[str, type] | None = None,
    state_shapes: dict[str, tuple[int, ...]] | None = None,
) -> coremltools.models.MLModel:
    """input_shapes and output_names use the Architecture map's names, in the wrapper's argument and return order.
    input_data_types gives the numpy type of each input that isn't float32 (integer inputs are np.int32).
    state_shapes names the wrapper's buffers that become Core ML state (MLState), which the model keeps between
    predictions; Core ML states are always float16 (watchOS 11).

    compute_precision is baked into the package. compute_units only applies to the copy returned here;
    whoever loads the saved package (test.py, the Swift app) chooses again."""
    input_data_types = input_data_types or {}
    states = [
        coremltools.StateType(wrapped_type=coremltools.TensorType(shape=shape, dtype=np.float16), name=name)
        for name, shape in (state_shapes or {}).items()
    ]
    return coremltools.convert(
        traced_wrapper,
        # float32 set explicitly: with a float16 model for watchOS 11, coremltools would default to float16.
        inputs=[
            coremltools.TensorType(name=name, shape=shape, dtype=input_data_types.get(name, np.float32))
            for name, shape in input_shapes.items()
        ],
        outputs=[coremltools.TensorType(name=name, dtype=np.float32) for name in output_names],
        states=states or None,
        convert_to="mlprogram",
        minimum_deployment_target=coremltools.target.watchOS11,
        compute_precision=compute_precision,
        compute_units=compute_units,
    )


def save_package(
    description: str,
    traced_wrapper,
    input_shapes: dict[str, tuple[int, ...]],
    output_names: list[str],
    model_path: Path,
    compute_precision,
    compute_units,
    input_data_types: dict[str, type] | None = None,
    state_shapes: dict[str, tuple[int, ...]] | None = None,
) -> coremltools.models.MLModel:
    """convert_to_coreml, then saves the package to model_path and prints its size. Returns the converted model."""
    print(f"[{description}]")
    coreml_model = convert_to_coreml(
        traced_wrapper, input_shapes, output_names, compute_precision, compute_units, input_data_types, state_shapes
    )
    model_path.parent.mkdir(exist_ok=True)
    coreml_model.save(str(model_path))
    print(f"  saved {model_path.relative_to(CONVERSION_DIRECTORY)} ({package_size_megabytes(model_path):.1f} MB)")
    return coreml_model


def print_compute_plans(coreml_model: coremltools.models.MLModel) -> None:
    """print_compute_plan for ALL, and for CPU_AND_NE, which is what Step 7's app loads (.cpuAndNeuralEngine). On the
    Mac the two can differ."""
    for compute_units in (coremltools.ComputeUnit.ALL, coremltools.ComputeUnit.CPU_AND_NE):
        print(f"[float16 model, {compute_units.name}: preferred chip per operation on this Mac]")
        print_compute_plan(coreml_model, compute_units)


def save_float32_and_float16_packages(
    traced_wrapper,
    input_shapes: dict[str, tuple[int, ...]],
    output_names: list[str],
    float32_model_path: Path,
    float16_model_path: Path,
    input_data_types: dict[str, type] | None = None,
    float16_package_input_data_types: dict[str, type] | None = None,
) -> None:
    """Converts and saves both packages, printing their sizes, then the float16 model's compute plans
    (print_compute_plans).
    input_data_types: as in convert_to_coreml, for both packages.
    float16_package_input_data_types: input types for the float16 package only, on top of input_data_types. Step 5
    takes k_cross and v_cross as np.float16 there, so Core ML doesn't convert them on every call; the float32 package
    keeps them float32, so its check stays exact."""
    input_data_types = input_data_types or {}
    float16_package_input_data_types = {**input_data_types, **(float16_package_input_data_types or {})}
    for description, model_path, compute_precision, compute_units, package_input_data_types in (
        ("float32, CPU only: proves the math", float32_model_path,
         coremltools.precision.FLOAT32, coremltools.ComputeUnit.CPU_ONLY, input_data_types),
        ("float16, all compute units: ships to the watch", float16_model_path,
         coremltools.precision.FLOAT16, coremltools.ComputeUnit.ALL, float16_package_input_data_types),
    ):
        coreml_model = save_package(
            description, traced_wrapper, input_shapes, output_names, model_path, compute_precision, compute_units,
            package_input_data_types,
        )
    print_compute_plans(coreml_model)


def print_compute_plan(coreml_model: coremltools.models.MLModel, compute_units: coremltools.ComputeUnit) -> None:
    """How many operations of each kind Core ML on this Mac puts on each chip, when compute_units are allowed.

    A preview only: the watch makes its own plan, which Step 7 checks with Instruments.
    """
    compute_plan = MLComputePlan.load_from_path(coreml_model.get_compiled_model_path(), compute_units=compute_units)
    operator_counts_by_device = collections.defaultdict(collections.Counter)
    for operation in compute_plan.model_structure.program.functions["main"].block.operations:
        device_usage = compute_plan.get_compute_device_usage_for_mlprogram_operation(operation)
        if operation.operator_name == "const" or device_usage is None:  # constants: nothing to run
            continue
        device_class_name = type(device_usage.preferred_compute_device).__name__  # e.g. MLNeuralEngineComputeDevice
        device_name = device_class_name.removeprefix("ML").removesuffix("ComputeDevice")
        operator_counts_by_device[device_name][operation.operator_name] += 1
    for device_name, operator_counts in operator_counts_by_device.items():
        counts_text = ", ".join(f"{count} {operator_name}" for operator_name, count in operator_counts.items())
        print(f"  {device_name}: {operator_counts.total()} ops ({counts_text})")


def package_size_megabytes(package_path: Path) -> float:
    return sum(file.stat().st_size for file in package_path.rglob("*") if file.is_file()) / 1e6
