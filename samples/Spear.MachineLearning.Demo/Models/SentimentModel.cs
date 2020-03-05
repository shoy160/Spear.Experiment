using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Spear.MachineLearning.Demo.Models
{
    /// <summary> 情绪预测模型 (二元分类)</summary>
    public class SentimentModel : ModelBuilder<SentimentData, SentimentPrediction>
    {
        public SentimentModel() : base("sentiment")
        {
        }

        protected override ITransformer TrainModel(IDataView dataView)
        {
            IEstimator<ITransformer> estimator =
                Context.Transforms.Text.FeaturizeText(FeaturesField, nameof(SentimentData.SentimentText))
                    .AppendCacheCheckpoint(Context);

            //添加训练算法
            estimator = estimator
                .Append(Context.BinaryClassification.Trainers.SdcaLogisticRegression()) //二元分类任务算法
                .Append(Context.BinaryClassification.Trainers.AveragedPerceptron(numberOfIterations: 10));

            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(dataView);
            Console.WriteLine("=============== End of training ===============");
            return model;
        }

        protected override void EvaluateModel(ITransformer model, IDataView dataView)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");

            var predictions = model.Transform(dataView);

            //二元分类模型评估
            var metrics = Context.BinaryClassification.Evaluate(predictions);

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }
    }

    public class SentimentData
    {
        [ColumnName("Label"), LoadColumn(0)]
        public bool Sentiment { get; set; }


        [LoadColumn(1)]
        public string SentimentText { get; set; }
    }

    public class SentimentPrediction : SentimentData
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }
        public float Score { get; set; }
    }
}
