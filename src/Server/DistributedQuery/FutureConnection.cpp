#include <Server/DistributedQuery/FutureConnection.h>
#include <Common/Exception.h>
#include <Common/logger_useful.h>
#include <base/scope_guard.h>
#include <fcntl.h>
#include <unistd.h>

#if defined(OS_LINUX)
#include <sys/eventfd.h>
#endif

namespace DB
{

namespace ErrorCodes
{
    extern const int CANNOT_OPEN_FILE;
    extern const int LOGICAL_ERROR;
}

FutureConnection::FutureConnection()
    : future(promise.get_future())
{
    createNotificationFd();
    LOG_TRACE(log, "Created FutureConnection");
}

FutureConnection::~FutureConnection()
{
    [[maybe_unused]] int err = close(notify_read_fd);
    chassert(!err || errno == EINTR);
    if (notify_write_fd != notify_read_fd)
    {
        err = close(notify_write_fd);
        chassert(!err || errno == EINTR);
    }
}

void FutureConnection::createNotificationFd()
{
#if defined(OS_LINUX)
    /// A single eventfd is both readable (for the poller) and writable (for the notifier).
    auto fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (fd == -1)
        throw Exception(ErrorCodes::CANNOT_OPEN_FILE, "Failed to create eventfd, error {}", errno);
    notify_read_fd = notify_write_fd = fd;
#else
    /// macOS has no eventfd; use a self-pipe. The read end is what the kqueue poller waits on.
    int fds[2];
    if (pipe(fds) == -1)
        throw Exception(ErrorCodes::CANNOT_OPEN_FILE, "Failed to create pipe, error {}", errno);
    for (int fd : fds)
    {
        if (fcntl(fd, F_SETFL, O_NONBLOCK) == -1 || fcntl(fd, F_SETFD, FD_CLOEXEC) == -1)
        {
            int saved_errno = errno;
            close(fds[0]);
            close(fds[1]);
            throw Exception(ErrorCodes::CANNOT_OPEN_FILE, "Failed to configure pipe, error {}", saved_errno);
        }
    }
    notify_read_fd = fds[0];
    notify_write_fd = fds[1];
#endif
}

int FutureConnection::getEventFd() const
{
    return notify_read_fd;
}

bool FutureConnection::isReady() const
{
    return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

Poco::Net::Socket FutureConnection::getSocket()
{
    if (!isReady())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "FutureConnection does not have a ready future, check is Ready() before calling getSocket()");

    // since it is a shared_future, multiple calls to get() are allowed and will return the same socket once it is set.
    return future.get();
}

void FutureConnection::setSocket(Poco::Net::Socket socket)
{
    /// First completion wins; a later setSocket/cancel is a no-op (the connection already paired
    /// or the query was torn down).
    if (satisfied.exchange(true))
        return;

    LOG_TRACE(log, "Setting socket for FutureConnection");
    promise.set_value(std::move(socket));
    notifyWaiter();
}

void FutureConnection::cancel(std::exception_ptr exception)
{
    if (satisfied.exchange(true))
        return;

    LOG_TRACE(log, "Cancelling FutureConnection");
    promise.set_exception(std::move(exception));
    notifyWaiter();
}

void FutureConnection::notifyWaiter() const
{
#if defined(OS_LINUX)
    uint64_t value = 1;
    constexpr ssize_t expected = sizeof(value);
#else
    /// A single byte is enough to make the self-pipe read end readable.
    char value = 1;
    constexpr ssize_t expected = sizeof(value);
#endif
    ssize_t written = 0;
    /// Retry on EINTR so a signal does not leave the promise ready while the poller is never woken.
    /// Other write failures cannot happen for a non-full, valid eventfd / self-pipe.
    do
        written = write(notify_write_fd, &value, sizeof(value));
    while (written < 0 && errno == EINTR);
    chassert(written == expected);
}

}
