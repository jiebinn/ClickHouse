#include <Storages/IStorage.h>
#include <Parsers/TablePropertiesQueriesASTs.h>
#include <Processors/Sources/SourceFromSingleChunk.h>
#include <QueryPipeline/BlockIO.h>
#include <DataTypes/DataTypeString.h>
#include <Columns/ColumnString.h>
#include <Common/typeid_cast.h>
#include <Access/Common/AccessFlags.h>
#include <Access/ContextAccess.h>
#include <Interpreters/Context.h>
#include <Interpreters/DatabaseCatalog.h>
#include <Interpreters/formatWithPossiblyHidingSecrets.h>
#include <Interpreters/InterpreterFactory.h>
#include <Interpreters/InterpreterShowCreateQuery.h>
#include <Interpreters/TableNameHints.h>
#include <Parsers/ASTCreateQuery.h>
#include <Core/Settings.h>
#include <Core/UUID.h>

namespace DB
{
namespace Setting
{
    extern const SettingsBool show_table_uuid_in_table_create_query_if_not_nil;
}

namespace ErrorCodes
{
    extern const int SYNTAX_ERROR;
    extern const int THERE_IS_NO_QUERY;
    extern const int BAD_ARGUMENTS;
    extern const int UNKNOWN_TABLE;
}

BlockIO InterpreterShowCreateQuery::execute()
{
    BlockIO res;
    res.pipeline = executeImpl();
    return res;
}


Block InterpreterShowCreateQuery::getSampleBlock()
{
    return Block{{
        ColumnString::create(),
        std::make_shared<DataTypeString>(),
        "statement"}};
}


QueryPipeline InterpreterShowCreateQuery::executeImpl()
{
    ASTPtr create_query;
    ASTQueryWithTableAndOutput * show_query = nullptr;
    if ((show_query = query_ptr->as<ASTShowCreateTableQuery>()) ||
        (show_query = query_ptr->as<ASTShowCreateViewQuery>()) ||
        (show_query = query_ptr->as<ASTShowCreateDictionaryQuery>()))
    {
        /// Only `SHOW CREATE TABLE` should resolve temporary tables for an unqualified name —
        /// `VIEW` and `DICTIONARY` cannot refer to a temporary table, so resolving to one would
        /// shadow a permanent view/dictionary with the same name and fail with `BAD_ARGUMENTS`.
        Context::StorageNamespace resolve_table_type = Context::ResolveOrdinary;
        if (show_query->isTemporary())
            resolve_table_type = Context::ResolveExternal;
        else if (query_ptr->as<ASTShowCreateTableQuery>())
            resolve_table_type = Context::ResolveAll;
        auto table_id = getContext()->resolveStorageID(*show_query, resolve_table_type);

        bool is_dictionary = static_cast<bool>(query_ptr->as<ASTShowCreateDictionaryQuery>());

        if (is_dictionary)
            getContext()->checkAccess(AccessType::SHOW_DICTIONARIES, table_id);
        else
            getContext()->checkAccess(AccessType::SHOW_COLUMNS, table_id);

        /// `SHOW CREATE DICTIONARY` is authorized with `SHOW DICTIONARIES`, which does not imply
        /// `SHOW TABLES`/`SHOW COLUMNS`. A user with only `SHOW DICTIONARIES` must not be able to tell
        /// a hidden regular table apart from a name that does not exist - otherwise `SHOW CREATE
        /// DICTIONARY` becomes an existence oracle for tables they may not see (the "is not a
        /// DICTIONARY" error below would confirm the table exists). So, for such a user, fetch and
        /// validate the *same* create query once: with no separate existence probe there is no TOCTOU
        /// window in which the object could be dropped and recreated as a different kind between the
        /// probe and the fetch. If the object is not a dictionary - a hidden table, a name that
        /// vanished, or one that fails to load - report it identically to a missing dictionary,
        /// carrying the same "Maybe you meant ...?" hint (which only ever suggests dictionaries to this
        /// user). Users who can also observe the object as a table keep the precise diagnostics below.
        /// This mirrors `InterpreterExistsQuery`, which likewise answers a dictionary-only user from a
        /// single observation instead of a second visibility-changing lookup.
        bool remask_as_missing_dictionary = false;
        if (is_dictionary)
        {
            const auto & access = getContext()->getAccess();
            const bool can_see_as_table
                = access->isGranted(AccessType::SHOW_TABLES, table_id.database_name, table_id.table_name)
                || access->isGranted(AccessType::SHOW_COLUMNS, table_id.database_name, table_id.table_name);
            if (!can_see_as_table)
            {
                try
                {
                    create_query = DatabaseCatalog::instance().getDatabase(table_id.database_name)->getCreateTableQuery(table_id.table_name, getContext());
                }
                catch (...)
                {
                    /// Ok to swallow: a missing name, or a hidden table still starting up or failing
                    /// to load, is treated as "not a dictionary". The error is not lost - it resurfaces
                    /// when a user who may see the object really accesses it.
                    create_query = nullptr;
                }
                if (!create_query || !create_query->as<ASTCreateQuery &>().is_dictionary)
                    remask_as_missing_dictionary = true;
            }
        }

        if (remask_as_missing_dictionary)
        {
            /// Discard any create query fetched for a hidden table so its definition cannot leak.
            create_query = nullptr;
            auto database = DatabaseCatalog::instance().getDatabase(table_id.database_name);
            TableNameHints hints(database, getContext());
            auto hint = hints.getHintForTable(table_id.table_name);
            if (hint.first.empty())
                throw Exception(ErrorCodes::UNKNOWN_TABLE, "There is no dictionary {}.{}",
                    backQuoteIfNeed(table_id.database_name), backQuoteIfNeed(table_id.table_name));
            throw Exception(ErrorCodes::UNKNOWN_TABLE, "There is no dictionary {}.{}. Maybe you meant {}.{}?",
                backQuoteIfNeed(table_id.database_name), backQuoteIfNeed(table_id.table_name),
                backQuoteIfNeed(hint.first), backQuoteIfNeed(hint.second));
        }

        if (!create_query)
            create_query = DatabaseCatalog::instance().getDatabase(table_id.database_name)->getCreateTableQuery(table_id.table_name, getContext());

        auto & ast_create_query = create_query->as<ASTCreateQuery &>();
        if (query_ptr->as<ASTShowCreateViewQuery>())
        {
            if (!ast_create_query.isView())
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "{}.{} is not a VIEW",
                    backQuote(ast_create_query.getDatabase()), backQuote(ast_create_query.getTable()));
        }
        else if (is_dictionary)
        {
            if (!ast_create_query.is_dictionary)
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "{}.{} is not a DICTIONARY",
                    backQuote(ast_create_query.getDatabase()), backQuote(ast_create_query.getTable()));
        }
    }
    else if ((show_query = query_ptr->as<ASTShowCreateDatabaseQuery>()))
    {
        if (show_query->isTemporary())
            throw Exception(ErrorCodes::SYNTAX_ERROR, "Temporary databases are not possible.");
        show_query->setDatabase(getContext()->resolveDatabase(show_query->getDatabase()));
        getContext()->checkAccess(AccessType::SHOW_DATABASES, show_query->getDatabase());
        create_query = DatabaseCatalog::instance().getDatabase(show_query->getDatabase())->getCreateDatabaseQuery();
    }

    if (!create_query)
        throw Exception(ErrorCodes::THERE_IS_NO_QUERY,
                        "Unable to show the create query of {}. Maybe it was created by the system.",
                        show_query->getTable());

    if (!getContext()->getSettingsRef()[Setting::show_table_uuid_in_table_create_query_if_not_nil])
    {
        auto & create = create_query->as<ASTCreateQuery &>();
        create.uuid = UUIDHelpers::Nil;
        if (create.targets)
            create.targets->resetInnerUUIDs();
    }

    MutableColumnPtr column = ColumnString::create();
    column->insert(format(
    {
        .ctx = getContext(),
        .query = *create_query,
        .one_line = false
    }));

    return QueryPipeline(std::make_shared<SourceFromSingleChunk>(std::make_shared<const Block>(Block{{
        std::move(column),
        std::make_shared<DataTypeString>(),
        "statement"}})));
}

void registerInterpreterShowCreateQuery(InterpreterFactory & factory);
void registerInterpreterShowCreateQuery(InterpreterFactory & factory)
{
    auto create_fn = [] (const InterpreterFactory::Arguments & args)
    {
        return std::make_unique<InterpreterShowCreateQuery>(args.query, args.context);
    };

    factory.registerInterpreter("InterpreterShowCreateQuery", create_fn);
}
}
