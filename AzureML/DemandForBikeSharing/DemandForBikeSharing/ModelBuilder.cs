using DemandForBikeSharing;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace BikeSharingDemand
{
    public class ModelBuilder
    {
        private readonly string trainingDataLocation;
        private readonly ILearningPipelineItem algorithm;

        public ModelBuilder(string trainingData, ILearningPipelineItem algo)
        {
            trainingDataLocation = trainingData;
            algorithm = algo;
        }

        public PredictionModel<BikeSharingDemandSample, BikeSharingDemandData> BuildAndTrain()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(trainingDataLocation).CreateFrom<BikeSharingDemandSample>(useHeader: true, separator: ','));
            pipeline.Add(new ColumnCopier(("Count", "Label")));
            pipeline.Add(new ColumnConcatenator("Features",
                                                "Season",
                                                "Year",
                                                "Month",
                                                "Hour",
                                                "Weekday",
                                                "Weather",
                                                "Temperature",
                                                "NormalizedTemperature",
                                                "Humidity",
                                                "Windspeed"));
            pipeline.Add(algorithm);

            return pipeline.Train<BikeSharingDemandSample, BikeSharingDemandData>();
        }
    }
}