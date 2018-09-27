using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SpamAnalysis
{
    class Program
    {
        static readonly string[] classNames = { "No Spam", "Spam" };

        static readonly IEnumerable<ClassificationData> predictClassData = new[]
        {
            new ClassificationData
            {
                Text = "Bust by this expressing at stepped and. My my dreary a and. Shaven we spoken minute beguiling my have gloated his fancy wandering back throws though. The chamber that rapping. So terrors is fast grim so of this grew from heard unto the land linking. Censer that and door the deep on word token and stayed. Door as the home maiden and gave surely some. Form sculptured soul quoth before both you whether for fact lost betook fowl meaninglittle implore the as unbroken the. A nodded quaff swung censer lenore tapping the morrow raven bird the. Hath lamplight the or beak expressing on remember little quoth. Disaster by mortals there that before. Dreary demons agreeing sinking thy denser mefilled visiter tapping land tossed unbroken with.</p><p>Came thy or thy the lenore reply perched bust that. From chamber i suddenly. It midnight some flown perched lamplight purple each forgiveness i of sat my lenore long the. Hesitating youhere then kind had lost then on of chamber respiterespite of. Is with murmured pondered chamber. He its door more i no tufted. Into the darkness and. By i streaming before nevermore as open dream i purple nevermore curious stayed before. My your so. That the lining his thrilled. With stately stepped in is an my then lenore that felt lies followed before oer i a thy pallid. That marvelled you whose shorn muttered more. A fowl stood feather.</p><p>Fearing core floor what heart. Ebony that and the as cushions unmerciful unhappy no i shadows prophet muttered that tossed at nothing. Word back i is he and shall his beating discourse still i ghost. Still angels be black is raven land oer though no. He at stopped that while there is a the the this what my take the chamber my spoken. Merely whom thy have stronger. Chamber parting name there bird scarcely home the into if. Nevermore hope uncertain each repeating kind press the the hear all that nothing the the. Open was at raven my. Before in discourse be be word. Token token and the i sign lining though volume of the tis quoth of shadows hesitating. Have what of and with more he in stock let door nothing bird quoth marvelled door this a. Sad whom faintly fowl i i explore upon out seraphim the one. Back flown nevermore croaking your the forget chamber wind some the some. The a mystery ominous above so heard lordly ah some days dreaming spoken cushions. For what nothing the but. And mefilled chamber whether there loneliness me your tempter if velvet."
            },
            new ClassificationData
            {
                Text = "So his chaste my. Mote way fabled as of aye from like old. Goodly rill near himnot den than to his none visit joyless. Shades climes that revellers had lyres by taste ways passed. To harold mote that earthly heralds at made sight of to a shamed once satiety left along he.</p><p>Heart he the evil she bower wassailers shades with one take loathed in of. Adieu he only shun come condole wight he he land these mammon shameless rhyme land. His the ancient he nor. Grief nor he nor amiss hour know by dear none like he. And to been a moths worse from his mother haply. Earth there fabled the they none his cared who labyrinth could sun reverie from fulness was so the. Thence birth fame where smile still and it and sacred far. Feere climes be sacred whilome from nor wins heavenly harold days unto the. Counsel harold a at. Flatterers eremites to. That and scape long and his it fall ofttimes condemned he sister lurked ne resolved olden. Most vile fellow where are he love he consecrate <a href=\"https://www.but.com\">muse</a> sorrow was. The break suits waste friend at one once.</p><p>Go <a href=\"https://www.sick.com\">are</a> which to but bower it things thy. Disappointed sad from true but are mirth on will below open mammon in mammon chill was. Sea that breast and other from earth the the of will kiss who venerable. A sadness of there and crime <a href=\"https://www.for.com\">like</a> and pile pillared. Suffice thy along did or had him none was vile this. Low sighed each lands. Once with a pleasure vaunted objects by strength and consecrate lines save the lowly bower before plain. Joyless fly he that. Condemned strange and known girls few grief found nor minstrels one he goodly her did glee fountain. Counsel scape from ive land and joyless before artless. Done nine to him."
            }
        };

        const string trainingDataFile = @"C:\DEVGIT\AzureML\AzureML\SpamAnalysis\SpamAnalysis\Data\training.tsv";
        const string testDataFile = @"C:\DEVGIT\AzureML\AzureML\SpamAnalysis\SpamAnalysis\Data\test.tsv";
        const string modelPath = @"C:\DEVGIT\AzureML\AzureML\SpamAnalysis\SpamAnalysis\Data\Model.zip";
        static void Main(string[] args)
        {
            Task.Run(async () =>
            {
                // Get a model trained to use for evaluation
                Console.WriteLine("Training Data Set");
                Console.WriteLine("-----------------");
                var trainModel = await TrainAsync(trainingDataFile, modelPath);

                Console.WriteLine();
                Console.WriteLine("Evaluating Training Results");
                Console.WriteLine("---------------------------");
                Evaluate(trainModel, testDataFile);

                var predictModel = await PredictAsync(modelPath, classNames, predictClassData);

                Console.WriteLine();
                Console.WriteLine("Please enter another string to classify or just <Enter> to exit the program.");

                var input = string.Empty;

                while (string.IsNullOrEmpty(input = Console.ReadLine()) == false)
                {
                    IEnumerable<ClassificationData> predictInputClass = new[]
                    {
                        new ClassificationData
                        {
                            Text = input
                        }
                    };

                    predictModel = await PredictAsync(modelPath, classNames, predictInputClass, predictModel);
                }

                Console.WriteLine("Press any key to end program...");
                Console.ReadKey();

            }).GetAwaiter().GetResult();
        }

        internal static async Task<PredictionModel<ClassificationData, ClassPrediction>>TrainAsync(string trainingDataFile, string modelPath)
        {
            // LearningPipeline allows you to add steps in order to keep everything together 
            // during the learning process.  
            var pipeline = new LearningPipeline();

            // The TextLoader loads a dataset with a class label and a corresponding text. 
            // When you create a loader, you specify the schema by passing a class to the loader containing
            // all the column names and their types. This is used to create the model, and train it. 
            pipeline.Add(new TextLoader(trainingDataFile).CreateFrom<ClassificationData>());

            // TextFeaturizer is a transform that is used to featurize an input column. 
            // This is used to format and clean the data.
            pipeline.Add(new TextFeaturizer("Features", "Text"));

            // Adds a FastTreeBinaryClassifier, the decision tree learner for this project, and 
            // three hyperparameters to be used for tuning decision tree performance.
            //pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });
            pipeline.Add(new StochasticGradientDescentBinaryClassifier());

            // Train the pipeline based on the dataset that has been loaded, transformed.
            PredictionModel<ClassificationData, ClassPrediction> model = pipeline.Train<ClassificationData, ClassPrediction>();

            // Saves the model we trained to a zip file.
            await model.WriteAsync(modelPath);

            // Returns the model we trained to use for evaluation.
            return model;
        }

        internal static void Evaluate(PredictionModel<ClassificationData, ClassPrediction> model, string testDatafile)
        {
            // loads the new test dataset with the same schema.
            // You can evaluate the model using this dataset as a quality check.

            var testData = new TextLoader(testDatafile).CreateFrom<ClassificationData>();

            // Computes the quality metrics for the PredictionModel using the specified dataset.
            var evaluator = new BinaryClassificationEvaluator();

            // The BinaryClassificationMetrics contains the overall metrics computed by binary
            // classification evaluators. To display these to determine the quality of the model,
            // you need to get the metrics first.
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            // Displaying the metrics for model validation
            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"     Auc: {metrics.Auc:P2}");
            Console.WriteLine($" F1Score: {metrics.F1Score:P2}");
        }

        internal static async Task<PredictionModel<ClassificationData, ClassPrediction>> PredictAsync(string modelPath, string[] classNames, IEnumerable<ClassificationData> predicts = null, PredictionModel<ClassificationData, ClassPrediction> model = null)
        {
            if (model == null)
            {
                model = await PredictionModel.ReadAsync<ClassificationData, ClassPrediction>(modelPath);
            }

            if (predicts == null) // do we have input to predict a result?
                return model;

            // Use the model to predict the class of the data.
            IEnumerable<ClassPrediction> predictions = model.Predict(predicts);

            Console.WriteLine();
            Console.WriteLine("Classification Predictions");
            Console.WriteLine("--------------------------");

            // Builds pairs of (input, prediction)
            IEnumerable<(ClassificationData input, ClassPrediction prediction)> inputsAndPredictions =
                predicts.Zip(predictions, (input, prediction) => (input, prediction));

            foreach (var item in inputsAndPredictions)
            {
                string textDisplay = item.input.Text;

                if (textDisplay.Length > 80)
                    textDisplay = textDisplay.Substring(0, 75) + "...";

                Console.WriteLine("Prediction: {0} | Text: '{1}'",
                          (item.prediction.Class ? classNames[0] : classNames[1]), textDisplay);
            }
            Console.WriteLine();

            return model;
        }
    }
}
