"""
Leitor de checkpoints PyTorch (.pth no formato zip) sem PyTorch.

Segurança: arquivos .pth são pickle, que pode executar código arbitrário. Este leitor só
aceita as poucas classes necessárias para reconstruir tensores e dicionários; qualquer
outra coisa no pickle é recusada com erro.
"""
from __future__ import annotations

import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

_STORAGE_DTYPES = {
    "FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64,
    "LongStorage": np.int64, "IntStorage": np.int32, "ByteStorage": np.uint8,
    "BFloat16Storage": None,  # tratado à parte
}


class _StorageType:
    def __init__(self, name: str):
        if name not in _STORAGE_DTYPES:
            raise pickle.UnpicklingError(f"tipo de storage não suportado: {name}")
        self.name = name


def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, hooks=None, metadata=None):
    if not size:
        return storage[offset:offset + 1].reshape(())
    itemsize = storage.itemsize
    return np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)).copy()


class _Unpickler(pickle.Unpickler):
    _ALLOWED = {
        ("collections", "OrderedDict"): OrderedDict,
        ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
        ("torch._utils", "_rebuild_parameter"): lambda data, requires_grad, hooks: data,
    }

    def __init__(self, fp, zf: zipfile.ZipFile, prefix: str):
        super().__init__(fp)
        self.zf, self.prefix, self.cache = zf, prefix, {}

    def find_class(self, module, name):
        if (module, name) in self._ALLOWED:
            return self._ALLOWED[(module, name)]
        if module == "torch" and name.endswith("Storage"):
            return _StorageType(name)
        raise pickle.UnpicklingError(f"classe não permitida no checkpoint: {module}.{name}")

    def persistent_load(self, pid):
        kind, storage_type, key, _location, _numel = pid
        if kind != "storage":
            raise pickle.UnpicklingError(f"persistent id inesperado: {kind}")
        if key not in self.cache:
            raw = self.zf.read(f"{self.prefix}/data/{key}")
            if storage_type.name == "BFloat16Storage":
                u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
                arr = u16.view(np.float32)
            else:
                arr = np.frombuffer(raw, dtype=np.dtype(_STORAGE_DTYPES[storage_type.name]).newbyteorder("<"))
            self.cache[key] = arr
        return self.cache[key]


def load_pth(path: str | Path) -> dict:
    """Retorna o conteúdo do checkpoint com tensores como arrays NumPy."""
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
        prefix = pkl[: -len("/data.pkl")]
        with zf.open(pkl) as fp:
            return _Unpickler(fp, zf, prefix).load()


def state_dict(path: str | Path) -> dict[str, np.ndarray]:
    """Extrai o state_dict, aceitando os formatos usados pelo BasicSR/Real-ESRGAN."""
    ckpt = load_pth(path)
    for k in ("params_ema", "params", "state_dict", "model"):
        if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
            ckpt = ckpt[k]
            break
    return {k.removeprefix("module."): np.asarray(v, dtype=np.float32) for k, v in ckpt.items()}
