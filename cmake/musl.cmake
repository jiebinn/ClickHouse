# Link musl libc built from sources (contrib/musl-cmake) into everything.
# The CRT startup files are wired into every executable's link line in
# cmake/linux/default_libs.cmake; see the ordering comment there.

target_link_libraries(global-libs INTERFACE musl)

# musl headers, in priority order: stubs for headers musl lacks (execinfo.h),
# generated headers, arch-specific bits/*.h, generic arch fallbacks, public headers.
target_include_directories(global-libs SYSTEM BEFORE INTERFACE
    "${ClickHouse_SOURCE_DIR}/contrib/musl-cmake/include"
    "${ClickHouse_BINARY_DIR}/contrib/musl-cmake/include"
    "${ClickHouse_SOURCE_DIR}/contrib/musl/arch/${MUSL_ARCH}"
    "${ClickHouse_SOURCE_DIR}/contrib/musl/arch/generic"
    "${ClickHouse_SOURCE_DIR}/contrib/musl/include"
)

# Kernel headers (linux/, asm/, asm-generic/) come from the glibc sysroot; the multiarch dir
# must be explicit because clang derives it from the *-linux-musl triple, not *-linux-gnu.
target_include_directories(global-libs SYSTEM INTERFACE
    "${CMAKE_SYSROOT}/usr/include"
    "${CMAKE_SYSROOT}/usr/include/${MUSL_ARCH}-linux-gnu"
)
