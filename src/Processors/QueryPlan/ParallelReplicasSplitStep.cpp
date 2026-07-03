#include <algorithm>
#include <Processors/QueryPlan/ParallelReplicasSplitStep.h>

namespace DB
{

void ParallelReplicasSplitStep::transformPipeline(QueryPipelineBuilder & /*pipeline*/, const BuildQueryPipelineSettings &)
{
    /// Pass-through when executed directly (no split was applied).
}

QueryPlanStepPtr ParallelReplicasSplitStep::clone() const
{
    return std::make_unique<ParallelReplicasSplitStep>(this->getOutputHeader(), context);
}

}
