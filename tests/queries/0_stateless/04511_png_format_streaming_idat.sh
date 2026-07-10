#!/usr/bin/env bash
# Tags: no-fasttest
# no-fasttest: the PNG format requires base64, and the fast-test build does not enable base64

# Regression test for the raw `FORMAT PNG` output path: the compressed pixel data must be emitted as a
# stream of bounded `IDAT` chunks, not materialized as one in-memory chunk. A large, incompressible image
# therefore produces several `IDAT` chunks, and peak memory stays bounded instead of growing with the image.
# See https://github.com/ClickHouse/ClickHouse/pull/110051.

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

OUT="${CLICKHOUSE_TMP}/04511_png_format_streaming_idat"
mkdir -p "${OUT}"

# An incompressible one-row image, wide enough that its compressed data spans more than one bounded
# `IDAT` chunk. Render over HTTP so the image is produced by the server-side encoder.
N=200000
QUERY="SELECT toUInt8(sipHash64(number) % 256) AS v FROM numbers(${N}) FORMAT PNG"
curl -sS "${CLICKHOUSE_URL}&output_format_image_width=${N}&output_format_image_height=1" \
    --data-binary "${QUERY}" > "${OUT}/big.png"

# The server's own pixel checksum, used to confirm the decoded image round-trips byte for byte.
EXPECTED_SUM=$(${CLICKHOUSE_CLIENT} --query "SELECT sum(toUInt8(sipHash64(number) % 256)) FROM numbers(${N})")

DECODER="${OUT}/decode.py"
cat > "${DECODER}" <<'PYEOF'
import sys, zlib, struct

path, expected_sum = sys.argv[1], int(sys.argv[2])
data = open(path, "rb").read()

print("signature ok" if data[:8] == b"\x89PNG\r\n\x1a\n" else "bad signature")

pos, idat, idat_chunks, width, height, color_type, crc_ok = 8, b"", 0, 0, 0, 0, True
while pos < len(data):
    length = struct.unpack(">I", data[pos:pos + 4])[0]
    ctype = data[pos + 4:pos + 8]
    chunk = data[pos + 8:pos + 8 + length]
    crc_stored = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])[0]
    if (zlib.crc32(ctype + chunk) & 0xffffffff) != crc_stored:
        crc_ok = False
    if ctype == b"IHDR":
        width, height, _bit_depth, color_type = struct.unpack(">IIBB", chunk[:10])
    elif ctype == b"IDAT":
        idat += chunk
        idat_chunks += 1
    elif ctype == b"IEND":
        break
    pos += 12 + length

print("all chunk crcs ok" if crc_ok else "crc mismatch")
print("multiple IDAT chunks: yes" if idat_chunks > 1 else "multiple IDAT chunks: no")

raw = zlib.decompress(idat)
channels = {0: 1, 2: 3, 6: 4}[color_type]
stride = width * channels
# All scanlines use the "None" filter (a leading 0 byte), so the pixels are the raw bytes.
filters_ok = all(raw[y * (stride + 1)] == 0 for y in range(height))
print("filters none ok" if filters_ok else "unexpected filter")

pixels = bytearray()
for y in range(height):
    pixels += raw[y * (stride + 1) + 1: (y + 1) * (stride + 1)]
print("pixels round-trip ok" if sum(pixels) == expected_sum else "pixels mismatch")
PYEOF

python3 "${DECODER}" "${OUT}/big.png" "${EXPECTED_SUM}"

rm -rf "${OUT}"
