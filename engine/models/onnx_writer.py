"""
Escritor mínimo de ONNX (protobuf à mão), sem depender dos pacotes `onnx` ou `torch`.

Motivo: o app precisa gerar/converter modelos em qualquer PC, inclusive os medianos,
sem instalar PyTorch (~2 GB). O conjunto de operadores necessário para as redes de SR
(Conv, PRelu, LeakyRelu, Add, Mul, Concat, DepthToSpace, Resize, Cast) é pequeno.

Números de campo seguem onnx/onnx.proto (ModelProto, GraphProto, NodeProto, TensorProto...).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

FLOAT, INT64, FLOAT16 = 1, 7, 10
_ATTR_FLOAT, _ATTR_INT, _ATTR_STRING, _ATTR_INTS = 1, 2, 3, 7


# ---------------- protobuf ----------------
def _varint(n: int) -> bytes:
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _int(field: int, v: int) -> bytes:
    return _key(field, 0) + _varint(v)


def _f32(field: int, v: float) -> bytes:
    return _key(field, 5) + struct.pack("<f", v)


def _bytes(field: int, b: bytes | str) -> bytes:
    if isinstance(b, str):
        b = b.encode()
    return _key(field, 2) + _varint(len(b)) + b


# ---------------- mensagens ONNX ----------------
def tensor(name: str, arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr)
    dtype = {np.float32: FLOAT, np.float16: FLOAT16, np.int64: INT64}[arr.dtype.type]
    return (b"".join(_int(1, d) for d in arr.shape) + _int(2, dtype) + _bytes(8, name)
            + _bytes(9, arr.astype(arr.dtype.newbyteorder("<")).tobytes()))


def value_info(name: str, elem: int, dims: list[int | str]) -> bytes:
    dim_msgs = b"".join(_bytes(1, _int(1, d) if isinstance(d, int) else _bytes(2, d)) for d in dims)
    tensor_type = _int(1, elem) + _bytes(2, dim_msgs)
    return _bytes(1, name) + _bytes(2, _bytes(1, tensor_type))


def attr(name: str, v) -> bytes:
    if isinstance(v, float):
        body = _bytes(1, name) + _int(20, _ATTR_FLOAT) + _f32(2, v)
    elif isinstance(v, int):
        body = _bytes(1, name) + _int(20, _ATTR_INT) + _int(3, v)
    elif isinstance(v, str):
        body = _bytes(1, name) + _int(20, _ATTR_STRING) + _bytes(4, v)
    else:
        body = _bytes(1, name) + _int(20, _ATTR_INTS) + b"".join(_int(8, i) for i in v)
    return _bytes(5, body)


def node(op: str, inputs: list[str], outputs: list[str], **attrs) -> bytes:
    return (b"".join(_bytes(1, i) for i in inputs) + b"".join(_bytes(2, o) for o in outputs)
            + _bytes(4, op) + b"".join(attr(k, v) for k, v in attrs.items()))


def model(nodes: list[bytes], inits: list[bytes], inputs: list[bytes], outputs: list[bytes],
          name: str = "graph", opset: int = 17, producer: str = "videoimprove") -> bytes:
    graph = (b"".join(_bytes(1, n) for n in nodes) + _bytes(2, name)
             + b"".join(_bytes(5, t) for t in inits)
             + b"".join(_bytes(11, i) for i in inputs) + b"".join(_bytes(12, o) for o in outputs))
    return _int(1, 8) + _bytes(2, producer) + _bytes(7, graph) + _bytes(8, _bytes(1, "") + _int(2, opset))


# ---------------- construtor de grafos ----------------
class GraphBuilder:
    """Constrói redes convolucionais NCHW. Com fp16=True, pesos e E/S ficam em float16."""

    def __init__(self, fp16: bool = False):
        self.fp16 = fp16
        self.dtype = np.float16 if fp16 else np.float32
        self.nodes: list[bytes] = []
        self.inits: list[bytes] = []
        self._n = 0
        self._consts: dict[float, str] = {}

    def _name(self, base: str) -> str:
        self._n += 1
        return f"{base}_{self._n}"

    def weight(self, name: str, arr: np.ndarray) -> str:
        self.inits.append(tensor(name, np.asarray(arr, dtype=self.dtype)))
        return name

    def const(self, v: float) -> str:
        if v not in self._consts:
            self._consts[v] = self.weight(self._name("const"), np.array([v]))
        return self._consts[v]

    def op(self, op: str, inputs: list[str], **attrs) -> str:
        out = self._name(op.lower())
        self.nodes.append(node(op, inputs, [out], **attrs))
        return out

    def conv(self, x: str, w: np.ndarray, b: np.ndarray, name: str) -> str:
        k = w.shape[-1]
        return self.op("Conv", [x, self.weight(name + ".weight", w), self.weight(name + ".bias", b)],
                       kernel_shape=[k, k], pads=[k // 2] * 4)

    def prelu(self, x: str, slope: np.ndarray, name: str) -> str:
        return self.op("PRelu", [x, self.weight(name, slope.reshape(-1, 1, 1))])

    def lrelu(self, x: str, alpha: float = 0.2) -> str:
        return self.op("LeakyRelu", [x], alpha=alpha)

    def add(self, a: str, b: str) -> str:
        return self.op("Add", [a, b])

    def scale(self, x: str, v: float) -> str:
        return self.op("Mul", [x, self.const(v)])

    def concat(self, xs: list[str]) -> str:
        return self.op("Concat", xs, axis=1)

    def pixel_shuffle(self, x: str, r: int) -> str:
        return self.op("DepthToSpace", [x], blocksize=r, mode="CRD")  # = torch.pixel_shuffle

    def upsample_nearest(self, x: str, r: int) -> str:
        key = f"scales_x{r}"
        if key not in self._consts:
            self.inits.append(tensor(key, np.array([1, 1, r, r], dtype=np.float32)))
            self._consts[key] = key
        return self.op("Resize", [x, "", key], mode="nearest")

    def save(self, path: str | Path, x: str, y: str) -> None:
        elem = FLOAT16 if self.fp16 else FLOAT
        self.nodes.append(node("Identity", [y], ["output"]))
        data = model(self.nodes, self.inits,
                     [value_info(x, elem, [1, 3, "height", "width"])],
                     [value_info("output", elem, [1, 3, "out_height", "out_width"])])
        Path(path).write_bytes(data)


def sr_model(path, scale: int, fp16: bool = False, broken: bool = False) -> None:
    """Modelo de teste: Resize nearest. broken=True divide por zero (simula FP16 estourando)."""
    g = GraphBuilder(fp16=False)
    elem = FLOAT16 if fp16 else FLOAT
    cur = "x"
    if fp16:
        cur = g.op("Cast", [cur], to=FLOAT)
    cur = g.upsample_nearest(cur, scale)
    if broken:
        cur = g.op("Div", [cur, g.const(0.0)])
    if fp16:
        cur = g.op("Cast", [cur], to=FLOAT16)
    g.nodes.append(node("Identity", [cur], ["y"]))
    Path(path).write_bytes(model(g.nodes, g.inits, [value_info("x", elem, [1, 3, "h", "w"])],
                                 [value_info("y", elem, [1, 3, "H", "W"])]))
