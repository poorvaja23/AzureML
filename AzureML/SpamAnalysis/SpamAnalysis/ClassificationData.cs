using Microsoft.ML.Runtime.Api;

namespace SpamAnalysis
{
    public class ClassificationData
    {
        [Column(ordinal: "0", name: "Label")]
        public float ClassLanel;

        [Column(ordinal: "1")]
        public string Text;
    }
}