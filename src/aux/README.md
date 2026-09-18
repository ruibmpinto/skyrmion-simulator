# Vendored third-party code

## libnpy

- Upstream: https://github.com/matajoh/libnpy
- Vendored version: 2.1.2 (see `libnpy/VERSION`)
- License: MIT (see `libnpy/LICENSE`)
- Integration: the source tree is committed directly, not a git
  submodule. `src/simulator_cpp/CMakeLists.txt` pulls it in with
  `add_subdirectory(... EXCLUDE_FROM_ALL)`.

The C++ simulator uses libnpy to write `.npz` archives that the Python
analysis scripts read back.

### Local patch: ZIP64 extra-field length

**This copy is patched and differs from upstream 2.1.2. Re-vendoring a
clean upstream tree silently reintroduces the bug.**

File: `libnpy/src/npz.cpp`, function `determine_extra_length`.

Upstream sums only the 8-byte ZIP64 fields (compressed size,
uncompressed size, and optionally the local-header offset) and writes
that sum as the local and central header `extra_field_length`. The value
it writes is 4 bytes short, because `write_zip64_extra` also emits a
2-byte tag and a 2-byte data-size ahead of those fields.

The patch adds the missing 4 bytes:

```cpp
if (length > 0) {
  length += 4;
}
```

Symptom without the patch: any `.npz` larger than roughly 2.25 GB, where
a late entry's offset exceeds `ZIP64_LIMIT`, is written with a malformed
ZIP64 extra record. `numpy.load` and Python's `zipfile` reject it with
`Corrupt extra field 0001`. Smaller archives never take the ZIP64 path
and are unaffected, which is why the bug only appears on large
trajectory dumps.

The reasoning is repeated as a comment at the patch site
(`libnpy/src/npz.cpp:114-124`).

### Re-vendoring checklist

1. Copy the new upstream tree over `libnpy/`.
2. Re-apply the `+= 4` in `determine_extra_length`, keeping the comment.
3. Update the version recorded above.
4. Verify with an `.npz` dump above 2.25 GB that `numpy.load` reads it
   back.
