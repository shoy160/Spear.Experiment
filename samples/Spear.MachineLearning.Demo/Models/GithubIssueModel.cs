using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Spear.MachineLearning.Demo.Models
{
    /// <summary>
    /// 问题分类 (多元分类)
    /// https://docs.microsoft.com/zh-cn/dotnet/machine-learning/tutorials/github-issue-classification
    /// </summary>
    public class GithubIssueModel : ModelBuilder<GitHubIssue, IssuePrediction>
    {
        public GithubIssueModel() : base("github_issue")
        {
        }

        protected override ITransformer TrainModel(IDataView dataView)
        {
            //数字键类型列
            IEstimator<ITransformer> estimator =
                Context.Transforms.Conversion.MapValueToKey(LabelField, nameof(GitHubIssue.Area));
            var fields = new[] { nameof(GitHubIssue.Title), nameof(GitHubIssue.Description) };
            var features = new List<string>();
            //特征列
            foreach (var field in fields)
            {
                var featureField = $"{field}Featurized";
                estimator = estimator.Append(Context.Transforms.Text.FeaturizeText(featureField, field));
                features.Add(featureField);
            }

            //合并特征列
            estimator = estimator.Append(Context.Transforms.Concatenate(FeaturesField, features.ToArray()));

            //缓存数据视图
            estimator = estimator.AppendCacheCheckpoint(Context);

            //添加训练算法
            estimator = estimator
                .Append(Context.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(Context.Transforms.Conversion.MapKeyToValue(PredictedLabelField));

            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(dataView);
            Console.WriteLine("=============== End of training ===============");
            return model;
        }

        protected override void EvaluateModel(ITransformer model, IDataView dataView)
        {
            //dataView = LoadFromText("issues_test.tsv");
            //模型评估
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            var predictions = model.Transform(dataView);

            //多元分类模型评估
            var metrics = Context.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine();
            Console.WriteLine(
                $"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine(
                $"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {metrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {metrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {metrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {metrics.LogLossReduction:#.###}");
            Console.WriteLine(
                $"*************************************************************************************************************");
        }
    }

    public class GitHubIssue
    {
        [LoadColumn(0)]
        public string ID { get; set; }

        [LoadColumn(1)]
        public string Area { get; set; }

        [LoadColumn(2)]
        public string Title { get; set; }

        [LoadColumn(3)]
        public string Description { get; set; }
    }

    public class IssuePrediction : GitHubIssue
    {
        [ColumnName("PredictedLabel")]
        public string PredictedArea { get; set; }
    }
}
