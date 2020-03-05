using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Spear.MachineLearning.Demo.Models
{
    /// <summary> 价格预测模型 (回归) </summary>
    public class TaxiTripModel : ModelBuilder<TaxiTrip, TaxiTripFarePrediction>
    {
        public TaxiTripModel() : base("taxi_trip")
        {
        }

        protected override ITransformer TrainModel(IDataView dataView)
        {
            IEstimator<ITransformer> estimator =
                Context.Transforms.CopyColumns(LabelField, nameof(TaxiTrip.FareAmount));

            //特征列
            var features = new List<string> { nameof(TaxiTrip.PassengerCount), nameof(TaxiTrip.TripDistance) };

            //值转换为数字
            var encoders = new[] { nameof(TaxiTrip.VendorId), nameof(TaxiTrip.RateCode), nameof(TaxiTrip.PaymentType) };
            foreach (var encoder in encoders)
            {
                var encoded = $"{encoder}Encoded";
                features.Add(encoded);
                estimator = estimator.Append(
                    Context.Transforms.Categorical.OneHotEncoding(encoded, encoder));
            }

            //合并特征列
            estimator = estimator.Append(Context.Transforms.Concatenate(FeaturesField, features.ToArray()));

            //学习算法
            estimator = estimator.Append(Context.Regression.Trainers.FastTree());

            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(dataView);
            Console.WriteLine("=============== End of Training ===============");
            return model;
        }

        protected override void EvaluateModel(ITransformer model, IDataView dataView)
        {
            //模型评估
            var predictions = model.Transform(dataView);

            //回归
            var metrics = Context.Regression.Evaluate(predictions);

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
        }
    }

    public class TaxiTrip
    {
        [LoadColumn(0)]
        public string VendorId { get; set; }

        [LoadColumn(1)]
        public string RateCode { get; set; }

        [LoadColumn(2)]
        public float PassengerCount { get; set; }

        [LoadColumn(3)]
        public float TripTime { get; set; }

        [LoadColumn(4)]
        public float TripDistance { get; set; }

        [LoadColumn(5)]
        public string PaymentType { get; set; }

        [LoadColumn(6)]
        public float FareAmount { get; set; }
    }

    public class TaxiTripFarePrediction : TaxiTrip
    {
        [ColumnName("Score")]
        public float PredictionAmount { get; set; }
    }
}
