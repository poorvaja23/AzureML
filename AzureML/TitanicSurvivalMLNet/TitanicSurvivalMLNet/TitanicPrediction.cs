using Microsoft.ML.Runtime.Api;

namespace TitanicSurvivalMLNet
{
    public class TitanicPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Survived;
    }
}