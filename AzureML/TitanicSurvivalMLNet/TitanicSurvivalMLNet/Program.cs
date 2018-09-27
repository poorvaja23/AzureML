using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TitanicSurvivalMLNet
{
    public class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string ModelPath => Path.Combine(AppPath, "TitanicModel.zip");
        public static void Main(string[] args)
        {
            MLforTitanicSurviors();
            
        }

        public static async Task MLforTitanicSurviors()
        {
            var model = await TrainAsync();
            Evaluate(model);

            var prediction = model.Predict(TestTitanicData.Passenger);
            Console.WriteLine($"Did this passenger survive?   Actual: Yes   Predicted: {(prediction.Survived ? "Yes" : "No")}");

            Console.ReadLine();
        }

        public static async Task<PredictionModel<TitanicData, TitanicPrediction>> TrainAsync()
        {
            // LearningPipeline holds all steps of the learning process: data, transforms, learners.  
            var pipeline = new LearningPipeline();

            // The TextLoader loads a dataset. The schema of the dataset is specified by passing a class containing
            // all the column names and their types.
            pipeline.Add(new TextLoader("C:\\DEVGIT\\AzureML\\AzureML\\TitanicSurvivalMLNet\\TitanicSurvivalMLNet\\titanic-train.csv").CreateFrom<TitanicData>(useHeader: true, separator: ','));

            // Transform any text feature to numeric values
            pipeline.Add(new CategoricalOneHotVectorizer(
                "Sex",
                "Ticket",
                "Fare",
                "Cabin",
                "Embarked"));

            // Put all features into a vector
            pipeline.Add(new ColumnConcatenator(
                "Features",
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Ticket",
                "Fare",
                "Cabin",
                "Embarked"));

            // FastTreeBinaryClassifier is an algorithm that will be used to train the model.
            // It has three hyperparameters for tuning decision tree performance. 
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });

            Console.WriteLine("=============== Training model ===============");
            // The pipeline is trained on the dataset that has been loaded and transformed.
            var model = pipeline.Train<TitanicData, TitanicPrediction>();

            // Saving the model as a .zip file.
            await model.WriteAsync(ModelPath);

            Console.WriteLine("=============== End training ===============");
            Console.WriteLine("The model is saved to {0}", ModelPath);

            return model;
        }

        private static void Evaluate(PredictionModel<TitanicData, TitanicPrediction> model)
        {
            var testData = new TextLoader("C:\\DEVGIT\\AzureML\\AzureML\\TitanicSurvivalMLNet\\TitanicSurvivalMLNet\\titanic-test.csv").CreateFrom<TitanicData>(useHeader: true, separator: ',');
            
            var evaluator = new BinaryClassificationEvaluator();

            Console.WriteLine("=============== Evaluating model ===============");

            var metrics = evaluator.Evaluate(model, testData);


            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End evaluating ===============");
            Console.WriteLine();
        }
    }

    internal class TestTitanicData
    {
        internal static readonly TitanicData Passenger = new TitanicData()
        {
            Pclass = 2,
            Name = "Brito, Mr. Amaury",
            Sex = "male",
            Age = 25,
            SibSp = 0,
            Parch = 1,
            Ticket = "230433",
            Fare = "26",
            Cabin = "",
            Embarked = "S"
        };
    }
}
