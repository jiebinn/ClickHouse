#pragma once

#include <Poco/Net/StreamSocket.h>
#include <Common/Logger.h>
#include <atomic>
#include <future>

namespace DB
{

/// Represents a connection that may not be established yet.
/// Provides a file descriptor that can be polled (epoll on Linux, kqueue on macOS) to wait
/// asynchronously for the connection: an `eventfd` on Linux, the read end of a self-pipe on macOS.
class FutureConnection
{
public:
    FutureConnection();
    ~FutureConnection();

    /// Get the notification file descriptor to register with the poller.
    int getEventFd() const;

    /// Check if the connection is ready (non-blocking)
    bool isReady() const;

    /// Try to get the socket
    /// Should only be called once the connection is ready, otherwise it will throw an exception.
    /// Could be called multiple times after connection is ready and will return the same socket.
    Poco::Net::Socket getSocket();

    /// Set the socket value (called when connection is established)
    /// Should be called only once, subsequent calls will throw an exception.
    void setSocket(Poco::Net::Socket socket);

    /// Wake the waiter with an exception. Used to cancel a still-pending
    /// connection (e.g. when the owning query is being torn down).
    /// At-most-once like `setSocket`.
    void cancel(std::exception_ptr exception);

private:
    void createNotificationFd();

    /// Wake the poller via the notification fd after the promise is completed.
    void notifyWaiter() const;

    std::promise<Poco::Net::Socket> promise;
    std::shared_future<Poco::Net::Socket> future;
    /// Guards the single allowed promise completion: setSocket and cancel race (the peer connecting
    /// vs. query teardown), and the loser must be a no-op rather than throw "promise already set".
    std::atomic<bool> satisfied{false};
    /// The pollable read side. On Linux this is an eventfd (read == write). On macOS it is the read
    /// end of a self-pipe; `notify_write_fd` is the write end used to wake the poller.
    int notify_read_fd = -1;
    int notify_write_fd = -1;
    LoggerPtr log = getLogger("FutureConnection");
};

using FutureConnectionPtr = std::shared_ptr<FutureConnection>;

}
