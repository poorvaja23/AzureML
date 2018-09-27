using Microsoft.ML.Runtime.Api;

namespace SpamAnalysis
{
    public class ClassPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Class;
    }
}