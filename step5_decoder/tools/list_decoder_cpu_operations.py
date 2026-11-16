"""Which of decoder_step.mlpackage's operations Core ML runs on the CPU, and what each one reads and writes.

Instruments on the Series 9 (2026-11-12) showed about 2.6 ms of every 20 ms decoder step in a Core ML conversion on
the CPU (E5RT::Ops::CastOperation), most of it plain memory copying (_platform_memmove). Copying that much each step
means copying something big, and the only big inputs are the cross K/V buffers, k_cross and v_cross (2.9 MB each).
This lists every operation that runs on the CPU, then every operation that reads k_cross or v_cross on any chip, each
with the names, types and shapes it reads and writes, to see whether a conversion of those buffers is among them.
Nothing is saved. Run from conversion/:

    python -m step5_decoder.tools.list_decoder_cpu_operations

The compute plan is this Mac's, for CPU_AND_NE (what the app loads). The watch makes its own, so this is a lead to
check against the watch, not proof.

What it showed (2026-11-12, coremltools 9.0, with the cross K/V still 40 wide): the 22 CPU operations are all small (the
masks, the rotary and token lookups, the scores' cast to float32), and k_cross and v_cross are read only by 12
slice_by_index on the Neural Engine. So the conversion wasn't in the program: Core ML added it, re-laying out the
40-wide inputs for the Neural Engine. Padding them to 64 removed it (common.config.DECODER_PADDED_HEAD_DIM).
"""

import coremltools
from coremltools.models.compute_plan import MLComputePlan
from coremltools.proto import MIL_pb2

from common.config import MODELS_DIRECTORY
from common.coreml_conversion import print_compute_plan

DECODER_STEP_PATH = MODELS_DIRECTORY / "decoder_step.mlpackage"  # what the app ships
CROSS_KEY_VALUE_INPUT_NAMES = {"k_cross", "v_cross"}


def main() -> None:
    compute_units = coremltools.ComputeUnit.CPU_AND_NE
    decoder_step = coremltools.models.MLModel(str(DECODER_STEP_PATH), compute_units=compute_units)
    value_types_by_name = program_value_types_by_name(decoder_step.get_spec())
    compute_plan = MLComputePlan.load_from_path(decoder_step.get_compiled_model_path(), compute_units=compute_units)

    print(f"[{DECODER_STEP_PATH.name}, {compute_units.name}: operations per chip on this Mac]")
    print_compute_plan(decoder_step, compute_units)

    operations_with_devices = []
    for operation in compute_plan.model_structure.program.functions["main"].block.operations:
        device_usage = compute_plan.get_compute_device_usage_for_mlprogram_operation(operation)
        if operation.operator_name == "const" or device_usage is None:  # constants: nothing to run
            continue
        device_class_name = type(device_usage.preferred_compute_device).__name__  # e.g. MLCPUComputeDevice
        operations_with_devices.append((operation, device_class_name.removeprefix("ML").removesuffix("ComputeDevice")))

    print("\n[every operation on the CPU, in program order]")
    for operation, device_name in operations_with_devices:
        if device_name == "CPU":
            print_operation(operation, device_name, compute_plan, value_types_by_name)

    print("\n[every operation that reads k_cross or v_cross, on any chip]")
    for operation, device_name in operations_with_devices:
        if CROSS_KEY_VALUE_INPUT_NAMES & set(input_names(operation)):
            print_operation(operation, device_name, compute_plan, value_types_by_name)


def program_value_types_by_name(spec) -> dict:
    """Each value's type in the package's program, by name: the model's inputs, then every operation's outputs."""
    main_function = spec.mlProgram.functions["main"]
    value_types_by_name = {named_value.name: named_value.type for named_value in main_function.inputs}
    for operation in main_function.block_specializations[main_function.opset].operations:
        for output in operation.outputs:
            value_types_by_name[output.name] = output.type
    return value_types_by_name


def input_names(operation) -> list[str]:
    """The named values an operation reads; constants written into the operation itself have no name."""
    return [binding.name for argument in operation.inputs.values() for binding in argument.bindings if binding.name]


def print_operation(operation, device_name: str, compute_plan: MLComputePlan, value_types_by_name: dict) -> None:
    cost = compute_plan.get_estimated_cost_for_mlprogram_operation(operation)
    cost_text = f"estimated cost {cost.weight:.4f}" if cost else "no cost estimate"
    print(f"  {operation.operator_name} on {device_name}, {cost_text}")
    for name in input_names(operation):
        print(f"      in   {name}: {type_text(value_types_by_name.get(name))}")
    for output in operation.outputs:
        print(f"      out  {output.name}: {type_text(value_types_by_name.get(output.name))}")


def type_text(value_type) -> str:
    """For example "FLOAT16 [6, 1, 8, 768, 40]"."""
    if value_type is None:
        return "(not in the package's program: Core ML renamed or added it when compiling)"
    kind = value_type.WhichOneof("type")
    if kind == "tensorType":
        tensor_type = value_type.tensorType
        shape = [
            dimension.constant.size if dimension.WhichOneof("dimension") == "constant" else "?"
            for dimension in tensor_type.dimensions
        ]
        return f"{MIL_pb2.DataType.Name(tensor_type.dataType)} {shape}"
    if kind == "stateType":
        return f"state of {type_text(value_type.stateType.wrappedType)}"
    return kind


if __name__ == "__main__":
    main()
