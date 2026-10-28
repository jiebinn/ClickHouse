#include <DataTypes/DataTypeString.h>
#include <Storages/System/SystemTableSourceRegistry.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeMap.h>
#include <DataTypes/DataTypeFactory.h>
#include <Interpreters/ProcessList.h>
#include <Storages/System/StorageSystemProcesses.h>
#include <Interpreters/Context.h>
#include <Core/Settings.h>
#include <Interpreters/ProfileEventsExt.h>
#include <Interpreters/formatWithPossiblyHidingSecrets.h>
#include <Common/typeid_cast.h>
#include <Common/IPv6ToBinary.h>
#include <Columns/ColumnsNumber.h>
#include <Columns/ColumnArray.h>

#include <Poco/Net/SocketAddress.h>


namespace DB
{

ColumnsDescription StorageSystemProcesses::getColumnsDescription()
{
    auto low_cardinality_string = std::make_shared<DataTypeLowCardinality>(std::make_shared<DataTypeString>());

    auto description = ColumnsDescription
    {
        {"is_initial_query", std::make_shared<DataTypeUInt8>(), "Whether the query is initial. Possible values: 1 — an initial (top-level) query, 0 — a child query initiated by another query, including queries for distributed execution and internal subqueries."},

        {"user", std::make_shared<DataTypeString>(), "The user who made the query. Keep in mind that for distributed processing, queries are sent to remote servers under the default user. The field contains the username for a specific query, not for a query that this query initiated."},
        {"query_id", std::make_shared<DataTypeString>(), "Query ID, if defined."},
        {"address", DataTypeFactory::instance().get("IPv6"), "The IP address the query was made from. The same for distributed processing. To track where a distributed query was originally made from, look at system.processes on the query requestor server."},
        {"port", std::make_shared<DataTypeUInt16>(), "The client port the query was made from."},

        {"initial_user", std::make_shared<DataTypeString>(), "Name of the user who ran the initial query in the same query chain."},
        {"initial_query_id", std::make_shared<DataTypeString>(), "ID of the initial query in the same query chain."},
        {"initial_address", DataTypeFactory::instance().get("IPv6"), "IP address from which the initial query in the same query chain was launched."},
        {"initial_port", std::make_shared<DataTypeUInt16>(), "Client port from which the initial query in the same query chain was launched."},

        {"interface", getReportedClientInterfaceEnum(), "The interface which was used to send the query, as reported by the client. `Unknown` if the reported interface is not one this server recognizes."},

        {"os_user", std::make_shared<DataTypeString>(), "Operating system username who runs clickhouse-client."},
        {"client_hostname", std::make_shared<DataTypeString>(), "Hostname of the client machine where the clickhouse-client or another TCP client is run."},
        {"client_name", std::make_shared<DataTypeString>(), "The clickhouse-client or another TCP client name."},
        {"client_agent", low_cardinality_string, "The AI coding agent that invoked the client (e.g. `claude-code`, `cursor`), detected from environment variables. Empty if no agent was detected."},
        {"client_revision", std::make_shared<DataTypeUInt64>(), "Revision of the clickhouse-client or another TCP client."},
        {"client_version_major", std::make_shared<DataTypeUInt64>(), "Major version of the clickhouse-client or another TCP client."},
        {"client_version_minor", std::make_shared<DataTypeUInt64>(), "Minor version of the clickhouse-client or another TCP client."},
        {"client_version_patch", std::make_shared<DataTypeUInt64>(), "Patch component of the clickhouse-client or another TCP client version."},

        {"http_method", getClientHTTPMethodEnum(), "HTTP method that initiated the query. `UNKNOWN` if the query did not arrive over HTTP, or if the reported method is not one this server recognizes."},
        {"http_user_agent", std::make_shared<DataTypeString>(), "HTTP header UserAgent passed in the HTTP query."},
        {"http_referer", std::make_shared<DataTypeString>(), "HTTP header Referer passed in the HTTP query (contains an absolute or partial address of the page making the query)."},
        {"forwarded_for", std::make_shared<DataTypeString>(), "HTTP header X-Forwarded-For passed in the HTTP query."},

        {"quota_key", std::make_shared<DataTypeString>(), "The quota key specified in the quotas setting (see keyed)."},
        {"distributed_depth", std::make_shared<DataTypeUInt64>(), "The number of times query was retransmitted between server nodes internally."},

        {"elapsed", std::make_shared<DataTypeFloat64>(), "The time in seconds since request execution started."},
        {"is_cancelled", std::make_shared<DataTypeUInt8>(), "Was query cancelled."},
        {"is_all_data_sent", std::make_shared<DataTypeUInt8>(), "Was all data sent to the client (in other words query had been finished on the server)."},
        {"read_rows", std::make_shared<DataTypeUInt64>(), "The number of rows read from the table. For distributed processing, on the requestor server, this is the total for all remote servers."},
        {"read_bytes", std::make_shared<DataTypeUInt64>(), "The number of uncompressed bytes read from the table. For distributed processing, on the requestor server, this is the total for all remote servers."},
        {"total_rows_approx", std::make_shared<DataTypeUInt64>(), "The approximation of the total number of rows that should be read. For distributed processing, on the requestor server, this is the total for all remote servers. It can be updated during request processing, when new sources to process become known."},
        {"written_rows", std::make_shared<DataTypeUInt64>(), "The amount of rows written to the storage."},
        {"written_bytes", std::make_shared<DataTypeUInt64>(), "The amount of bytes written to the storage."},
        {"memory_usage", std::make_shared<DataTypeInt64>(), "Amount of RAM the query uses. It might not include some types of dedicated memory"},
        {"peak_memory_usage", std::make_shared<DataTypeInt64>(), "The current peak of memory usage."},
        {"spillable_memory_bytes", std::make_shared<DataTypeInt64>(), "The current spillable memory."},
        {"query", std::make_shared<DataTypeString>(), "The query text. For INSERT, it does not include the data to insert."},
        {"normalized_query_hash", std::make_shared<DataTypeUInt64>(), "A numeric hash value, such as it is identical for queries differ only by values of literals."},
        {"query_kind", std::make_shared<DataTypeString>(), "The type of the query - SELECT, INSERT, etc."},

        {"thread_ids", std::make_shared<DataTypeArray>(std::make_shared<DataTypeUInt64>()), "The list of identifiers of all threads which participated in this query."},
        {"peak_threads_usage", std::make_shared<DataTypeUInt64>(), "Maximum count of simultaneous threads executing the query."},
        {"ProfileEvents", std::make_shared<DataTypeMap>(low_cardinality_string, std::make_shared<DataTypeUInt64>()), "ProfileEvents calculated for this query."},
        {"Settings", std::make_shared<DataTypeMap>(low_cardinality_string, low_cardinality_string), "The list of modified user-level settings."},

        {"current_database", std::make_shared<DataTypeString>(), "The name of the current database."},

        {"is_internal", std::make_shared<DataTypeUInt8>(), "Indicates whether it is an auxiliary query executed internally."},
    };

    description.setAliases({
        {"ProfileEvents.Names", {std::make_shared<DataTypeArray>(std::make_shared<DataTypeString>())}, "mapKeys(ProfileEvents)"},
        {"ProfileEvents.Values", {std::make_shared<DataTypeArray>(std::make_shared<DataTypeUInt64>())}, "mapValues(ProfileEvents)"},
        {"Settings.Names", {std::make_shared<DataTypeArray>(std::make_shared<DataTypeString>())}, "mapKeys(Settings)" },
        {"Settings.Values", {std::make_shared<DataTypeArray>(std::make_shared<DataTypeString>())}, "mapValues(Settings)"}
    });

    return description;
}

void StorageSystemProcesses::fillData(MutableColumns & res_columns, ContextPtr context, const ActionsDAG::Node *, std::vector<UInt8> columns_mask) const
{
    ProcessList::Info info = context->getProcessList().getInfo(columns_mask[38] || columns_mask[39], columns_mask[40], columns_mask[41] || columns_mask[42]);
    const bool show_secrets = columns_mask[41] && canDisplaySecrets(context);

    for (const auto & process : info)
    {
        size_t i = 0;

        if (columns_mask[0])
            res_columns[i++]->insert(process.client_info.query_kind == ClientInfo::QueryKind::INITIAL_QUERY);

        if (columns_mask[1])
            res_columns[i++]->insert(process.client_info.current_user);
        if (columns_mask[2])
            res_columns[i++]->insert(process.client_info.current_query_id);
        if (columns_mask[3])
            res_columns[i++]->insertData(IPv6ToBinary(process.client_info.current_address->host()).data(), 16);
        if (columns_mask[4])
            res_columns[i++]->insert(process.client_info.current_address->port());

        if (columns_mask[5])
            res_columns[i++]->insert(process.client_info.initial_user);
        if (columns_mask[6])
            res_columns[i++]->insert(process.client_info.initial_query_id);
        if (columns_mask[7])
            res_columns[i++]->insertData(IPv6ToBinary(process.client_info.initial_address->host()).data(), 16);
        if (columns_mask[8])
            res_columns[i++]->insert(process.client_info.initial_address->port());

        if (columns_mask[9])
            res_columns[i++]->insert(reportedClientInterfaceEnumValue(process.client_info.interface));

        if (columns_mask[10])
            res_columns[i++]->insert(process.client_info.os_user);
        if (columns_mask[11])
            res_columns[i++]->insert(process.client_info.getClientHostName());
        if (columns_mask[12])
            res_columns[i++]->insert(process.client_info.client_name);
        if (columns_mask[13])
            res_columns[i++]->insert(process.client_info.client_agent);
        if (columns_mask[14])
            res_columns[i++]->insert(process.client_info.client_tcp_protocol_version);
        if (columns_mask[15])
            res_columns[i++]->insert(process.client_info.client_version_major);
        if (columns_mask[16])
            res_columns[i++]->insert(process.client_info.client_version_minor);
        if (columns_mask[17])
            res_columns[i++]->insert(process.client_info.client_version_patch);

        if (columns_mask[18])
            res_columns[i++]->insert(reportedClientHTTPMethodEnumValue(process.client_info.http_method));
        if (columns_mask[19])
            res_columns[i++]->insert(process.client_info.http_user_agent);
        if (columns_mask[20])
            res_columns[i++]->insert(process.client_info.http_referer);
        if (columns_mask[21])
            res_columns[i++]->insert(process.client_info.forwarded_for);

        if (columns_mask[22])
            res_columns[i++]->insert(process.client_info.quota_key);
        if (columns_mask[23])
            res_columns[i++]->insert(process.client_info.distributed_depth);

        if (columns_mask[24])
            res_columns[i++]->insert(static_cast<double>(process.elapsed_microseconds) / 1'000'000.0);
        if (columns_mask[25])
            res_columns[i++]->insert(process.is_cancelled);
        if (columns_mask[26])
            res_columns[i++]->insert(process.is_all_data_sent);
        if (columns_mask[27])
            res_columns[i++]->insert(process.read_rows);
        if (columns_mask[28])
            res_columns[i++]->insert(process.read_bytes);
        if (columns_mask[29])
            res_columns[i++]->insert(process.total_rows);
        if (columns_mask[30])
            res_columns[i++]->insert(process.written_rows);
        if (columns_mask[31])
            res_columns[i++]->insert(process.written_bytes);
        if (columns_mask[32])
            res_columns[i++]->insert(process.memory_usage);
        if (columns_mask[33])
            res_columns[i++]->insert(process.peak_memory_usage);
        if (columns_mask[34])
            res_columns[i++]->insert(process.spillable_memory_bytes);
        if (columns_mask[35])
            res_columns[i++]->insert(process.query);
        if (columns_mask[36])
            res_columns[i++]->insert(process.normalized_query_hash);
        if (columns_mask[37])
            res_columns[i++]->insert(magic_enum::enum_name(process.query_kind));

        if (columns_mask[38])
        {
            Array threads_array;
            threads_array.reserve(process.thread_ids.size());
            for (const UInt64 thread_id : process.thread_ids)
                threads_array.emplace_back(thread_id);
            res_columns[i++]->insert(threads_array);
        }

        if (columns_mask[39])
            res_columns[i++]->insert(process.peak_threads_usage);

        if (columns_mask[40])
        {
            IColumn * column = res_columns[i++].get();

            if (process.profile_counters)
                ProfileEvents::dumpToMapColumn(*process.profile_counters, column, true);
            else
            {
                column->insertDefault();
            }
        }

        if (columns_mask[41])
        {
            IColumn * column = res_columns[i++].get();

            if (process.query_settings)
                process.query_settings->dumpToMapColumn(column, true, show_secrets);
            else
            {
                column->insertDefault();
            }
        }

        if (columns_mask[42])
            res_columns[i++]->insert(process.current_database);
        if (columns_mask[43])
            res_columns[i++]->insert(process.is_internal);
    }
}

}

/// Register the source file of this system table for `system.documentation`.
namespace DB { REGISTER_SYSTEM_TABLE_SOURCE(StorageSystemProcesses) }
