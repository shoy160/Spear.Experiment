using System;
using System.Linq;
using Newtonsoft.Json;
using Spear.MachineLearning.Demo.Models;

namespace Spear.MachineLearning.Demo
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            //            SentimentTest();
            //            GithubIssueTest();
            //            TaxiTripTest();
            //MovieRecommendTest();
            Console.WriteLine(new DateTime(2020, 2, 29).AddYears(1).ToString("yyyy-MM-dd HH:mm:ss"));
        }

        private static void SentimentTest()
        {
            var builder = new SentimentModel();

            var dataView = builder.LoadFromText("comment.tsv", allowQuoting: true);

            builder.Train(dataView);

            //var inputs = new List<SentimentData>();

            while (true)
            {
                var word = Console.ReadLine();
                if (string.Equals(word, "exit", StringComparison.CurrentCultureIgnoreCase))
                    break;

                //inputs.Add(new SentimentData
                //{
                //    SentimentText = word
                //});
                //if (inputs.Count < 3) continue;

                //var results = builder.Predict(inputs);
                //foreach (var result in results)
                //{
                //    Console.WriteLine($"Text\t\t:{result.SentimentText}");

                //    Console.WriteLine($"Prediction\t:{result.Prediction}");
                //    Console.WriteLine($"Probability\t:{result.Probability}");
                //    Console.WriteLine($"Score\t\t:{result.Score}");
                //}
                //inputs.Clear();

                var result = builder.Predict(new SentimentData
                {
                    SentimentText = word
                });

                Console.WriteLine($"Text\t\t:{result.SentimentText}");

                Console.WriteLine($"Prediction\t:{result.Prediction}");
                Console.WriteLine($"Probability\t:{result.Probability}");
                Console.WriteLine($"Score\t\t:{result.Score}");
            }
        }

        private static void GithubIssueTest()
        {
            var model = new GithubIssueModel();
            //var dataView = model.LoadFromText("issues_train.tsv");
            //model.Train(dataView);

            var data = model.LoadFromText("issues_test.tsv");
            model.Evaluate(data);
            var testData = model.Data.CreateEnumerable<GitHubIssue>(data, false).ToArray();
            while (true)
            {
                var word = Console.ReadLine();
                if (string.Equals(word, "exit", StringComparison.CurrentCultureIgnoreCase))
                    break;
                var item = testData.OrderBy(t => new Random().Next()).First();

                var output = model.Predict(item);

                Console.WriteLine(JsonConvert.SerializeObject(new
                {
                    output.ID,
                    output.Title,
                    output.Area,
                    output.PredictedArea
                }, Formatting.Indented));
            }
        }

        private static void TaxiTripTest()
        {
            var model = new TaxiTripModel();

            ////模型训练
            //var dataView = model.LoadFromText("taxi-fare-train.csv", ',');
            //model.Train(dataView);

            var data = model.LoadFromText("taxi-fare-test.csv", ',');
            model.Evaluate(data);

            var testData = model.Data.CreateEnumerable<TaxiTrip>(data, false).ToArray();
            while (true)
            {
                var word = Console.ReadLine();
                if (string.Equals(word, "exit", StringComparison.CurrentCultureIgnoreCase))
                    break;
                var item = testData.OrderBy(t => new Random().Next()).First();

                var output = model.Predict(item);

                Console.WriteLine(JsonConvert.SerializeObject(new
                {
                    output.VendorId,
                    output.RateCode,
                    output.PassengerCount,
                    output.TripTime,
                    output.TripDistance,
                    output.PaymentType,
                    output.FareAmount,
                    output.PredictionAmount
                }, Formatting.Indented));
            }
        }

        private static void MovieRecommendTest()
        {
            var model = new MovieRecommendModel();

            ////模型训练
            //var dataView = model.LoadFromText("recommendation-ratings-train.csv", ',');
            //model.Train(dataView);

            var data = model.LoadFromText("recommendation-ratings-test.csv", ',');
            model.Evaluate(data);

            var testData = model.Data.CreateEnumerable<MovieRating>(data, false).ToArray();
            while (true)
            {
                var word = Console.ReadLine();
                if (string.Equals(word, "exit", StringComparison.CurrentCultureIgnoreCase))
                    break;
                var item = testData.OrderBy(t => new Random().Next()).First();

                var output = model.Predict(item);

                if (Math.Round(output.Score, 1) > 3.5)
                {
                    Console.WriteLine("Movie " + item.MovieId + " is recommended for user " + item.UserId);
                }
                else
                {
                    Console.WriteLine("Movie " + item.MovieId + " is not recommended for user " + item.UserId);
                }
                Console.WriteLine(JsonConvert.SerializeObject(new
                {
                    userId = output.UserId,
                    movieId = output.MovieId,
                    output.Label,
                    output.Score
                }, Formatting.Indented));

            }
        }
    }
}
