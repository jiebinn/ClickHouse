if (NOT ENABLE_LIBPQXX)
    return()
endif()

set(LIBPQ_SOURCE_DIR "${ClickHouse_SOURCE_DIR}/contrib/libpq")

set(SRCS
        "${LIBPQ_SOURCE_DIR}/fe-auth.c"
        "${LIBPQ_SOURCE_DIR}/fe-auth-scram.c"
        "${LIBPQ_SOURCE_DIR}/fe-connect.c"
        "${LIBPQ_SOURCE_DIR}/fe-exec.c"
        "${LIBPQ_SOURCE_DIR}/fe-lobj.c"
        "${LIBPQ_SOURCE_DIR}/fe-misc.c"
        "${LIBPQ_SOURCE_DIR}/fe-print.c"
        "${LIBPQ_SOURCE_DIR}/fe-trace.c"
        "${LIBPQ_SOURCE_DIR}/fe-protocol3.c"
        "${LIBPQ_SOURCE_DIR}/fe-secure.c"
        "${LIBPQ_SOURCE_DIR}/fe-secure-common.c"
        "${LIBPQ_SOURCE_DIR}/fe-secure-openssl.c"
        "${LIBPQ_SOURCE_DIR}/legacy-pqsignal.c"
        "${LIBPQ_SOURCE_DIR}/libpq-events.c"
        "${LIBPQ_SOURCE_DIR}/pqexpbuffer.c"

        "${LIBPQ_SOURCE_DIR}/common/scram-common.c"
        "${LIBPQ_SOURCE_DIR}/common/sha2.c"
        "${LIBPQ_SOURCE_DIR}/common/sha1.c"
        "${LIBPQ_SOURCE_DIR}/common/md5.c"
        "${LIBPQ_SOURCE_DIR}/common/md5_common.c"
        "${LIBPQ_SOURCE_DIR}/common/hmac_openssl.c"
        "${LIBPQ_SOURCE_DIR}/common/cryptohash.c"
        "${LIBPQ_SOURCE_DIR}/common/saslprep.c"
        "${LIBPQ_SOURCE_DIR}/common/unicode_norm.c"
        "${LIBPQ_SOURCE_DIR}/common/ip.c"
        "${LIBPQ_SOURCE_DIR}/common/jsonapi.c"
        "${LIBPQ_SOURCE_DIR}/common/wchar.c"
        "${LIBPQ_SOURCE_DIR}/common/base64.c"
        "${LIBPQ_SOURCE_DIR}/common/link-canary.c"
        "${LIBPQ_SOURCE_DIR}/common/fe_memutils.c"
        "${LIBPQ_SOURCE_DIR}/common/string.c"
        "${LIBPQ_SOURCE_DIR}/common/pg_get_line.c"
        "${LIBPQ_SOURCE_DIR}/common/stringinfo.c"
        "${LIBPQ_SOURCE_DIR}/common/psprintf.c"
        "${LIBPQ_SOURCE_DIR}/common/encnames.c"
        "${LIBPQ_SOURCE_DIR}/common/logging.c"

        "${LIBPQ_SOURCE_DIR}/port/snprintf.c"
        "${LIBPQ_SOURCE_DIR}/port/strlcpy.c"
        "${LIBPQ_SOURCE_DIR}/port/strerror.c"
        "${LIBPQ_SOURCE_DIR}/port/inet_net_ntop.c"
        "${LIBPQ_SOURCE_DIR}/port/getpeereid.c"
        "${LIBPQ_SOURCE_DIR}/port/chklocale.c"
        "${LIBPQ_SOURCE_DIR}/port/noblock.c"
        "${LIBPQ_SOURCE_DIR}/port/pg_strong_random.c"
        "${LIBPQ_SOURCE_DIR}/port/pgstrcasecmp.c"
        "${LIBPQ_SOURCE_DIR}/port/thread.c"
        "${LIBPQ_SOURCE_DIR}/port/path.c"
        )

add_library(_libpq ${SRCS})

add_definitions(-DHAVE_BIO_METH_NEW)
add_definitions(-DHAVE_HMAC_CTX_NEW)
add_definitions(-DHAVE_HMAC_CTX_FREE)

target_include_directories (_libpq SYSTEM PUBLIC ${LIBPQ_SOURCE_DIR})
target_include_directories (_libpq SYSTEM PUBLIC "${LIBPQ_SOURCE_DIR}/include")
target_include_directories (_libpq SYSTEM PRIVATE "${LIBPQ_SOURCE_DIR}/configs")

# NOTE: this is a dirty hack to avoid and instead pg_config.h should be shipped
# for different OS'es like for jemalloc, not one generic for all OS'es like
# now.
if (OS_DARWIN OR OS_FREEBSD OR USE_MUSL)
    target_compile_definitions(_libpq PRIVATE -DSTRERROR_R_INT=1)
endif()

target_link_libraries (_libpq PRIVATE OpenSSL::SSL)

add_library(ch_contrib::libpq ALIAS _libpq)
