#include <Backups/IBackupCoordination.h>
#include <Backups/BackupFileInfo.h>


namespace DB
{

void IBackupCoordination::forEachFileInfoForAllHosts(const std::function<void(const BackupFileInfo &)> & callback) const
{
    for (const auto & file_info : getFileInfosForAllHosts())
        callback(file_info);
}

}
