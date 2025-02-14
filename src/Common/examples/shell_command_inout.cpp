#include <thread>

#include <Common/ShellCommand.h>
#include <Common/Exception.h>

#include <IO/ReadBufferFromFileDescriptor.h>
#include <IO/WriteBufferFromFileDescriptor.h>
#include <IO/copyData.h>
#include <iostream>

/** This example shows how we can proxy stdin to ShellCommand and obtain stdout in streaming fashion. */

int main(int argc, char ** argv)
try
{
    using namespace DB;

    if (argc < 2)
    {
        std::cerr << "Usage: shell_command_inout 'command...' < in > out\n";
        return 1;
    }

    auto command = ShellCommand::execute(argv[1]);

    ReadBufferFromFileDescriptor in(STDIN_FILENO);
    WriteBufferFromFileDescriptor out(STDOUT_FILENO);
    WriteBufferFromFileDescriptor err(STDERR_FILENO);

    /// Background thread sends data and foreground thread receives result.

    std::thread thread([&]
    {
        copyData(in, command->in);
        command->in.close();
    });

    copyData(command->out, out);
    out.finalize();

    copyData(command->err, err);
    err.finalize();

    thread.join();
    return 0;
}
catch (...)
{
    std::cerr << DB::getCurrentExceptionMessage(true) << '\n';
    throw;
}
