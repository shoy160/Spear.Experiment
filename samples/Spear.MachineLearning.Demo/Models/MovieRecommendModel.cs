using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Spear.MachineLearning.Demo.Models
{
    /// <summary> 影片推荐模型 (矩阵因子分析) </summary>
    public class MovieRecommendModel : ModelBuilder<MovieRating, MovieRatingPrediction>
    {
        public MovieRecommendModel() : base("movie_recommend")
        {
        }

        protected override ITransformer TrainModel(IDataView dataView)
        {
            IEstimator<ITransformer> estimator =
                Context.Transforms.Conversion.MapValueToKey("userIdEncoded", nameof(MovieRating.UserId))
                    .Append(Context.Transforms.Conversion.MapValueToKey("movieIdEncoded", nameof(MovieRating.MovieId)));

            //训练算法 矩阵分解
            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "movieIdEncoded",
                LabelColumnName = LabelField,
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            estimator = estimator.Append(Context.Recommendation().Trainers.MatrixFactorization(options));

            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(dataView);
            Console.WriteLine("=============== End of Training ===============");

            return model;
        }

        protected override void EvaluateModel(ITransformer model, IDataView dataView)
        {
            Console.WriteLine("=============== Evaluating the model ===============");
            var prediction = model.Transform(dataView);
            var metrics = Context.Regression.Evaluate(prediction);

            Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError);
            Console.WriteLine("RSquared: " + metrics.RSquared);
        }
    }

    public class MovieRating
    {
        [LoadColumn(0)]
        public float UserId;

        [LoadColumn(1)]
        public float MovieId;

        [LoadColumn(2)]
        public float Label;
    }

    public class MovieRatingPrediction : MovieRating
    {
        public float Score;
    }
}
