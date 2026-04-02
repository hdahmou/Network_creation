"""Test-only compatibility shims."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, cast

import zarr.api.synchronous as zarr_sync_api
import zarr.codecs._v2 as zarr_v2_codecs
import zarr.codecs.numcodecs._codecs as zarr_numcodecs
import zarr.core.array as zarr_array
import zarr.core.group as zarr_group
import zarr.core.sync as zarr_core_sync
import zarr.core.sync_group as zarr_sync_group
from zarr.core.buffer import Buffer
from zarr.core.buffer.core import default_buffer_prototype
from zarr.storage._local import LocalStore, _get, _put


_PATCHED = False


def _patch_zarr_for_tests() -> None:
    global _PATCHED
    if _PATCHED:
        return

    original_sync = zarr_core_sync.sync

    def _sync_without_background_loop(
        coro: Coroutine[Any, Any, Any],
        loop: asyncio.AbstractEventLoop | None = None,
        timeout: float | None = None,
    ) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        return original_sync(coro, loop=loop, timeout=timeout)

    async def _set_without_to_thread(
        self: LocalStore,
        key: str,
        value: Buffer,
        exclusive: bool = False,
    ) -> None:
        if not self._is_open:
            await self._open()
        self._check_writable()
        if exclusive and (self.root / key).exists():
            raise FileExistsError(self.root / key)
        _put(self.root / key, value, exclusive=False)
        await asyncio.sleep(0)

    async def _get_without_to_thread(
        self: LocalStore,
        key: str,
        prototype: Any = None,
        byte_range: Any = None,
    ) -> Buffer | None:
        if prototype is None:
            prototype = default_buffer_prototype()
        if not self._is_open:
            await self._open()
        path = self.root / key
        try:
            return _get(path, prototype, byte_range)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

    async def _get_partial_values_without_to_thread(
        self: LocalStore,
        prototype: Any,
        key_ranges: Any,
    ) -> list[Buffer | None]:
        values: list[Buffer | None] = []
        for key, byte_range in key_ranges:
            values.append(await _get_without_to_thread(self, key, prototype, byte_range))
        return values

    async def _delete_without_to_thread(self: LocalStore, key: str) -> None:
        self._check_writable()
        path = self.root / key
        if path.is_dir():
            import shutil

            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    async def _exists_without_to_thread(self: LocalStore, key: str) -> bool:
        return (self.root / key).is_file()

    async def _bytes_bytes_decode_without_to_thread(
        self: Any,
        chunk_data: Buffer,
        chunk_spec: Any,
    ) -> Buffer:
        return zarr_numcodecs.as_numpy_array_wrapper(
            self._codec.decode,
            chunk_data,
            chunk_spec.prototype,
        )

    async def _bytes_bytes_encode_without_to_thread(
        self: Any,
        chunk_data: Buffer,
        chunk_spec: Any,
    ) -> Buffer:
        return self._encode(chunk_data, chunk_spec.prototype)

    async def _array_array_decode_without_to_thread(
        self: Any,
        chunk_data: Any,
        chunk_spec: Any,
    ) -> Any:
        out = self._codec.decode(chunk_data.as_ndarray_like())
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

    async def _array_array_encode_without_to_thread(
        self: Any,
        chunk_data: Any,
        chunk_spec: Any,
    ) -> Any:
        out = self._codec.encode(chunk_data.as_ndarray_like())
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out)

    async def _array_bytes_decode_without_to_thread(
        self: Any,
        chunk_data: Buffer,
        chunk_spec: Any,
    ) -> Any:
        out = self._codec.decode(chunk_data.to_bytes())
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

    async def _array_bytes_encode_without_to_thread(
        self: Any,
        chunk_data: Any,
        chunk_spec: Any,
    ) -> Buffer:
        out = self._codec.encode(chunk_data.as_ndarray_like())
        return chunk_spec.prototype.buffer.from_bytes(out)

    async def _v2_decode_without_to_thread(
        self: Any,
        chunk_bytes: Buffer,
        chunk_spec: Any,
    ) -> Any:
        cdata = chunk_bytes.as_array_like()
        chunk = self.compressor.decode(cdata) if self.compressor else cdata
        if self.filters:
            for codec in reversed(self.filters):
                chunk = codec.decode(chunk)
        chunk = zarr_v2_codecs.ensure_ndarray_like(chunk)
        if chunk_spec.dtype.dtype_cls is not zarr_v2_codecs.np.dtypes.ObjectDType:
            try:
                chunk = chunk.view(chunk_spec.dtype.to_native_dtype())
            except TypeError:
                chunk = zarr_v2_codecs.np.array(chunk).astype(chunk_spec.dtype.to_native_dtype())
        elif chunk.dtype != object:
            raise RuntimeError("cannot read object array without object codec")
        chunk = chunk.reshape(-1, order="A")
        chunk = chunk.reshape(chunk_spec.shape, order=chunk_spec.order)
        return zarr_v2_codecs.get_ndbuffer_class().from_ndarray_like(chunk)

    async def _v2_encode_without_to_thread(
        self: Any,
        chunk_array: Any,
        chunk_spec: Any,
    ) -> Buffer | None:
        chunk = chunk_array.as_ndarray_like()
        chunk = chunk.astype(chunk_spec.dtype.to_native_dtype(), order=chunk_spec.order, copy=False)
        if self.filters:
            for codec in self.filters:
                chunk = codec.encode(chunk)
        if zarr_v2_codecs.ensure_ndarray_like(chunk).dtype == object:
            raise RuntimeError("cannot write object array without object codec")
        cdata = self.compressor.encode(chunk) if self.compressor else chunk
        return chunk_spec.prototype.buffer.from_bytes(zarr_v2_codecs.ensure_bytes(cdata))

    zarr_core_sync.sync = _sync_without_background_loop
    zarr_sync_api.sync = _sync_without_background_loop
    zarr_array.sync = _sync_without_background_loop
    zarr_group.sync = _sync_without_background_loop
    zarr_sync_group.sync = _sync_without_background_loop
    zarr_numcodecs._NumcodecsBytesBytesCodec._decode_single = cast(
        Any,
        _bytes_bytes_decode_without_to_thread,
    )
    zarr_numcodecs._NumcodecsBytesBytesCodec._encode_single = cast(
        Any,
        _bytes_bytes_encode_without_to_thread,
    )
    zarr_numcodecs._NumcodecsArrayArrayCodec._decode_single = cast(
        Any,
        _array_array_decode_without_to_thread,
    )
    zarr_numcodecs._NumcodecsArrayArrayCodec._encode_single = cast(
        Any,
        _array_array_encode_without_to_thread,
    )
    zarr_numcodecs._NumcodecsArrayBytesCodec._decode_single = cast(
        Any,
        _array_bytes_decode_without_to_thread,
    )
    zarr_numcodecs._NumcodecsArrayBytesCodec._encode_single = cast(
        Any,
        _array_bytes_encode_without_to_thread,
    )
    zarr_v2_codecs.V2Codec._decode_single = cast(Any, _v2_decode_without_to_thread)
    zarr_v2_codecs.V2Codec._encode_single = cast(Any, _v2_encode_without_to_thread)
    LocalStore.get = cast(Any, _get_without_to_thread)
    LocalStore.get_partial_values = cast(Any, _get_partial_values_without_to_thread)
    LocalStore._set = cast(Any, _set_without_to_thread)
    LocalStore.delete = cast(Any, _delete_without_to_thread)
    LocalStore.exists = cast(Any, _exists_without_to_thread)
    _PATCHED = True


_patch_zarr_for_tests()
