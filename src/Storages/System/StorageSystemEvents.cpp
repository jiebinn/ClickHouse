#include <Columns/IColumn.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnsNumber.h>
#include <Storages/System/SystemTableSourceRegistry.h>
#include <Common/ProfileEvents.h>
#include <Core/Settings.h>
#include <Interpreters/Context.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Storages/System/StorageSystemEvents.h>
#include <Storages/VirtualColumnUtils.h>
#include <Common/assert_cast.h>

namespace DB
{
namespace Setting
{
    extern const SettingsBool system_events_show_zero_values;
}

ColumnsDescription StorageSystemEvents::getColumnsDescription()
{
    auto description = ColumnsDescription
    {
        {"event", std::make_shared<DataTypeString>(), "Event name."},
        {"value", std::make_shared<DataTypeUInt64>(), "Number of events occurred."},
        {"description", std::make_shared<DataTypeString>(), "Event description."},
    };

    description.setAliases({
        {"name", std::make_shared<DataTypeString>(), "event"}
    });

    return description;
}

Block StorageSystemEvents::getFilterSampleBlock() const
{
    return {{{}, std::make_shared<DataTypeString>(), "event"}};
}

void StorageSystemEvents::fillData(MutableColumns & res_columns, ContextPtr context, const ActionsDAG::Node * predicate, std::vector<UInt8>) const
{
    const bool show_zero_values = context->getSettingsRef()[Setting::system_events_show_zero_values];
    auto add_event = [&](ProfileEvents::Event event)
    {
        UInt64 value = ProfileEvents::global_counters[event];
        if (value != 0 || show_zero_values)
        {
            res_columns[0]->insert(ProfileEvents::getName(event));
            res_columns[1]->insert(value);
            res_columns[2]->insert(ProfileEvents::getDocumentation(event));
        }
    };

    if (predicate)
    {
        auto event_names = ColumnString::create();
        auto event_ids = ColumnUInt64::create();
        for (ProfileEvents::Event event = ProfileEvents::Event(0), end = ProfileEvents::end(); event < end; ++event)
        {
            event_names->insert(ProfileEvents::getName(event));
            event_ids->insertValue(event.toUnderType());
        }

        Block events_block
        {
            {std::move(event_names), std::make_shared<DataTypeString>(), "event"},
            {std::move(event_ids), std::make_shared<DataTypeUInt64>(), "_event_id"},
        };
        VirtualColumnUtils::filterBlockWithPredicate(predicate, events_block, context);
        const auto & selected_events = assert_cast<const ColumnUInt64 &>(*events_block.getByName("_event_id").column).getData();
        for (const auto event : selected_events)
            add_event(ProfileEvents::Event(event));
    }
    else
    {
        for (ProfileEvents::Event event = ProfileEvents::Event(0), end = ProfileEvents::end(); event < end; ++event)
            add_event(event);
    }
}

}

/// Register the source file of this system table for `system.documentation`.
namespace DB { REGISTER_SYSTEM_TABLE_SOURCE(StorageSystemEvents) }
