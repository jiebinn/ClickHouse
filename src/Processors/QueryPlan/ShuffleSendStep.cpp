#include <Processors/QueryPlan/ShuffleSendStep.h>
#include <Processors/QueryPlan/QueryPlanStepRegistry.h>
#include <Processors/Sinks/NativeCompressedSink.h>
#include <Processors/QueryPlan/Serialization.h>
#include <Processors/QueryPlan/IParameterLookup.h>
#include <Processors/QueryPlan/ExchangeLookup.h>
#include <Processors/QueryPlan/LogicalExchangeStep.h>
#include <Processors/Transforms/ScatterByPartitionTransform.h>
#include <Processors/ResizeProcessor.h>
#include <Processors/Port.h>
#include <QueryPipeline/QueryPipelineBuilder.h>
#include <QueryPipeline/Pipe.h>
#include <IO/WriteHelpers.h>
#include <IO/ReadHelpers.h>
#include <Core/ColumnNumbers.h>
#include <DataTypes/DataTypeFactory.h>

#include <vector>

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
    extern const int NOT_IMPLEMENTED;
}

QueryPipelineBuilderPtr ShuffleSendStep::updatePipeline(QueryPipelineBuilders pipelines, const BuildQueryPipelineSettings & settings)
{
    /// K upstream streams -> K * ScatterByPartitionTransform(1 -> M)
    ///                    -> M * ResizeProcessor(K -> 1) -> M sinks; K = getNumStreams, M = num_buckets.
    auto & pipeline = *pipelines.front();
    auto stream_header = pipeline.getSharedHeader();

    /// Totals/extremes have no defined shuffle semantics; throw rather than silently lose them.
    if (pipeline.hasTotals() || pipeline.hasExtremes())
        throw Exception(ErrorCodes::NOT_IMPLEMENTED, "ShuffleSendStep does not support pipelines with totals or extremes");

    ColumnNumbers key_columns;
    key_columns.reserve(key_names.size());
    for (const auto & key_name : key_names)
        key_columns.push_back(stream_header->getPositionByName(key_name));

    const size_t num_streams = pipeline.getNumStreams();
    chassert(num_streams > 0);

    /// Both layers are built in one transform call so that the intermediate num_streams * num_buckets
    /// port count never becomes the pipe's max_parallel_streams (it would inflate the executor thread limit).
    pipeline.transform([&](OutputPortRawPtrs ports)
    {
        if (ports.size() != num_streams)
            throw Exception(ErrorCodes::LOGICAL_ERROR,
                "ShuffleSendStep: expected {} output ports, got {}", num_streams, ports.size());

        Processors result;

        /// One scatter per upstream stream; output b of scatter k carries the rows of bucket b from stream k.
        std::vector<std::vector<OutputPort *>> scatter_outputs(num_streams);
        for (size_t stream = 0; stream < num_streams; ++stream)
        {
            auto scatter = std::make_shared<ScatterByPartitionTransform>(stream_header, num_buckets, key_columns, hash_cast_types);
            connect(*ports[stream], scatter->getInputs().front());
            scatter_outputs[stream].reserve(num_buckets);
            for (auto & output : scatter->getOutputs())
                scatter_outputs[stream].push_back(&output);
            result.push_back(std::move(scatter));
        }

        /// For a single stream the scatter alone already produces num_buckets ports in bucket order.
        if (num_streams == 1)
            return result;

        /// Merge the num_streams ports of each bucket into one with a ResizeProcessor.
        for (size_t bucket = 0; bucket < num_buckets; ++bucket)
        {
            auto resize = std::make_shared<ResizeProcessor>(stream_header, num_streams, 1);
            auto input_it = resize->getInputs().begin();
            for (size_t stream = 0; stream < num_streams; ++stream, ++input_it)
                connect(*scatter_outputs[stream][bucket], *input_it);
            result.push_back(std::move(resize));
        }

        return result;
    });

    chassert(pipeline.getNumStreams() == num_buckets);

    const String shard_id = settings.parameter_lookup->getParameter("bucket_id").safeGet<String>();

    size_t bucket = 0;
    pipeline.setSinks([&](const SharedHeader & header, Pipe::StreamType stream_type)
    {
        chassert(stream_type == Pipe::StreamType::Main);
        String destination_bucket_id = toString(bucket);
        ++bucket;
        return settings.exchange_lookup->createSink(header, ExchangeStreamId(exchange_id, shard_id, destination_bucket_id));
    });

    if (bucket != num_buckets)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "ShuffleSendStep: expected {} buckets, but created only {}", num_buckets, bucket);

    return std::move(pipelines.front());
}

namespace
{

void serializeNames(const Names & names, WriteBuffer & out)
{
    writeVarUInt(names.size(), out);
    for (const String & name : names)
        writeStringBinary(name, out);
}

void deserializeNames(Names & names, ReadBuffer & in)
{
    size_t size = 0;
    readVarUInt(size, in);
    names.resize(size);
    for (size_t i = 0; i < size; ++i)
        readStringBinary(names[i], in);
}

}

void ShuffleSendStep::serialize(Serialization & ctx) const
{
    writeStringBinary(exchange_id, ctx.out);
    serializeNames(key_names, ctx.out);
    writeVarUInt(num_buckets, ctx.out);

    writeVarUInt(hash_cast_types.size(), ctx.out);
    for (const auto & type : hash_cast_types)
        writeStringBinary(type ? type->getName() : "", ctx.out);
}

std::unique_ptr<IQueryPlanStep> ShuffleSendStep::deserialize(Deserialization & ctx)
{
    String exchange_id;
    readStringBinary(exchange_id, ctx.in);

    Names key_names;
    deserializeNames(key_names, ctx.in);

    size_t num_buckets = 0;
    readVarUInt(num_buckets, ctx.in);

    size_t hash_cast_count = 0;
    readVarUInt(hash_cast_count, ctx.in);
    DataTypes hash_cast_types;
    hash_cast_types.reserve(hash_cast_count);
    for (size_t i = 0; i < hash_cast_count; ++i)
    {
        String type_name;
        readStringBinary(type_name, ctx.in);
        hash_cast_types.push_back(type_name.empty() ? nullptr : DataTypeFactory::instance().get(type_name));
    }

    return std::make_unique<ShuffleSendStep>(ctx.input_headers.front(), exchange_id, std::move(key_names), num_buckets, std::move(hash_cast_types));
}

void registerShuffleSendStep(QueryPlanStepRegistry & registry);
void registerShuffleSendStep(QueryPlanStepRegistry & registry)
{
    registry.registerStep("ShuffleSend", ShuffleSendStep::deserialize);
}

}
