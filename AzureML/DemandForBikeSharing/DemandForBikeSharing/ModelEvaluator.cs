using DemandForBikeSharing;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;

namespace BikeSharingDemand
{
    public class ModelEvaluator
    {
        public RegressionMetrics Evaluate(PredictionModel<BikeSharingDemandSample, BikeSharingDemandData> model, string testDataLocation)
        {
                var testData = new TextLoader(testDataLocation).CreateFrom<BikeSharingDemandSample>(useHeader: true, separator: ',');
                var metrics = new RegressionEvaluator().Evaluate(model, testData);
                return metrics;
        }
    }
}