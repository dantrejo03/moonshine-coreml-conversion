"""Reading Moonshine's .ort files as data: the stored graph and tensors, without running anything. For the tools that
check what a quantized .ort piece computes (step3_adapter/tools/inspect_position_table.py,
step4_cross_kv/tools/inspect_cross_kv_ort.py)."""

import numpy as np
from onnxruntime.tools.ort_format_model.ort_flatbuffers_py import fbs

from common.config import MOONSHINE_ORT_MODEL_DIRECTORY

TENSOR_DATA_TYPE_NAMES = {
    value: name for name, value in vars(fbs.TensorDataType.TensorDataType).items() if not name.startswith("_")
}
NUMPY_TYPES_BY_TENSOR_DATA_TYPE = {
    fbs.TensorDataType.TensorDataType.FLOAT: np.float32,
    fbs.TensorDataType.TensorDataType.UINT8: np.uint8,
    fbs.TensorDataType.TensorDataType.INT8: np.int8,
    fbs.TensorDataType.TensorDataType.INT32: np.int32,
    fbs.TensorDataType.TensorDataType.INT64: np.int64,
}
PRINTED_VALUES_MAXIMUM_COUNT = 8  # print_graph prints the values of stored tensors this small: shapes, axes, scalars


def load_ort_graph(file_name: str):
    """The graph stored in one of Moonshine's .ort files (a flatbuffer), for reading only."""
    ort_file_bytes = (MOONSHINE_ORT_MODEL_DIRECTORY / file_name).read_bytes()
    return fbs.InferenceSession.InferenceSession.GetRootAsInferenceSession(ort_file_bytes, 0).Model().Graph()


def stored_tensor_values(tensor) -> np.ndarray:
    """A stored tensor's values, with its own type and shape. Fails for types this project hasn't needed yet."""
    tensor_name = tensor.Name().decode()
    numpy_type = NUMPY_TYPES_BY_TENSOR_DATA_TYPE.get(tensor.DataType())
    if numpy_type is None:
        raise ValueError(f"{tensor_name}: no numpy type for {TENSOR_DATA_TYPE_NAMES.get(tensor.DataType())}")
    if tensor.RawDataIsNone():
        raise ValueError(f"{tensor_name}: values not stored inline in the graph")
    dimensions = [tensor.Dims(i) for i in range(tensor.DimsLength())]
    return np.frombuffer(tensor.RawDataAsNumpy().tobytes(), numpy_type).reshape(dimensions)


def print_graph(graph) -> dict:
    """Prints every stored tensor (initializer), with its values when it has at most PRINTED_VALUES_MAXIMUM_COUNT, and
    every node with its attributes (such as Transpose's perm). Returns the stored tensors by name."""
    initializers_by_name = {}
    print("  stored tensors:")
    for index in range(graph.InitializersLength()):
        tensor = graph.Initializers(index)
        name = tensor.Name().decode()
        initializers_by_name[name] = tensor
        dimensions = [tensor.Dims(i) for i in range(tensor.DimsLength())]
        values_text = ""
        if tensor.DataType() in NUMPY_TYPES_BY_TENSOR_DATA_TYPE and np.prod(dimensions) <= PRINTED_VALUES_MAXIMUM_COUNT:
            values_text = f" = {stored_tensor_values(tensor).tolist()}"
        data_type_name = TENSOR_DATA_TYPE_NAMES.get(tensor.DataType(), tensor.DataType())
        print(f"    {name}: {data_type_name} {dimensions}{values_text}")
    print("  nodes, in stored order:")
    for index in range(graph.NodesLength()):
        node = graph.Nodes(index)
        inputs = [node.Inputs(i).decode() for i in range(node.InputsLength())]
        outputs = [node.Outputs(i).decode() for i in range(node.OutputsLength())]
        attributes = [_attribute_text(node.Attributes(i)) for i in range(node.AttributesLength())]
        attributes_text = f"  [{', '.join(attributes)}]" if attributes else ""
        print(f"    {node.OpType().decode()}({', '.join(inputs)}) -> {', '.join(outputs)}{attributes_text}")
    return initializers_by_name


def _attribute_text(attribute) -> str:
    """name=value for number and text attributes; name=<TYPE> for the rest (tensors, subgraphs)."""
    attribute_types = fbs.AttributeType.AttributeType
    attribute_type = attribute.Type()
    if attribute_type == attribute_types.INT:
        value = attribute.I()
    elif attribute_type == attribute_types.INTS:
        value = [attribute.Ints(i) for i in range(attribute.IntsLength())]
    elif attribute_type == attribute_types.FLOAT:
        value = attribute.F()
    elif attribute_type == attribute_types.STRING:
        value = attribute.S().decode()
    else:
        type_names = {number: name for name, number in vars(attribute_types).items() if not name.startswith("_")}
        value = f"<{type_names.get(attribute_type, attribute_type)}>"
    return f"{attribute.Name().decode()}={value}"
