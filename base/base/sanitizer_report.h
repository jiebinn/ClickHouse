#pragma once

/// Captures sanitizer runtime output into a preallocated global buffer,
/// so that the core dump analyzer can read it.

#include <base/sanitizer_defs.h>

#ifdef SANITIZER
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-identifier"
#pragma clang diagnostic ignored "-Wmissing-prototypes"

extern "C"
{
char sanitizer_report[1 << 20];
unsigned long sanitizer_report_size = 0;
}

static char sanitizer_report_lock;

static DISABLE_SANITIZER_INSTRUMENTATION void appendToSanitizerReport(const char * str)
{
    unsigned long i = sanitizer_report_size;
    while (*str != '\0' && i < sizeof(sanitizer_report) - 1)
        sanitizer_report[i++] = *str++;
    sanitizer_report_size = i;
}

extern "C" DISABLE_SANITIZER_INSTRUMENTATION void __sanitizer_on_print(const char * str)
{
    /// Writing to sanitizer_report_size by previous thread must happen-before reading from sanitizer_report_size by this thread.
    /// Hence, we need acquire-release.
    while (__atomic_test_and_set(&sanitizer_report_lock, __ATOMIC_ACQUIRE))
        ;

    /// The preamble makes the buffer discoverable by scanning the core dump.
    /// It is assembled from parts so its only full copy is in this buffer.
    if (sanitizer_report_size == 0)
    {
        appendToSanitizerReport("CLICKHOUSE");
        appendToSanitizerReport(" SANITIZER");
        appendToSanitizerReport(" REPORT\n");
    }
    appendToSanitizerReport(str);

    __atomic_clear(&sanitizer_report_lock, __ATOMIC_RELEASE);
}

#pragma clang diagnostic pop
#endif
