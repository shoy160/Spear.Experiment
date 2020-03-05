using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;

namespace Spear.MachineLearning.Demo
{
    public abstract class ModelBuilder<TInput, TOutput>
        where TInput : class
        where TOutput : class, new()
    {
        private readonly string _modelPath;

        protected const string LabelField = "Label";
        protected const string FeaturesField = "Features";
        protected const string PredictedLabelField = "PredictedLabel";


        protected ModelBuilder(string name)
        {
            _modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", $"{name}_model.zip");
            Context = new MLContext(1);
        }

        protected MLContext Context { get; }

        public DataOperationsCatalog Data => Context.Data;

        public string CombinePath(string path)
        {
            return Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", path);
        }

        public IDataView LoadFromText(string path, char separator = '\t', bool header = true, bool allowQuoting = false)
        {
            if (!File.Exists(path))
                path = CombinePath(path);

            return Context.Data.LoadFromTextFile<TInput>(path, separator, header, allowQuoting);
        }

        /// <summary> 训练模型 </summary>
        /// <param name="dataView"></param>
        public void Train(IDataView dataView)
        {
            var testData = Context.Data.TrainTestSplit(dataView, 0.2);
            var model = TrainModel(testData.TrainSet);
            EvaluateModel(model, testData.TestSet);
            Context.Model.Save(model, dataView.Schema, _modelPath);
        }

        /// <summary> 模型训练 </summary>
        /// <param name="dataView"></param>
        /// <returns></returns>
        protected abstract ITransformer TrainModel(IDataView dataView);

        /// <summary> 模型评估 </summary>
        /// <param name="model"></param>
        /// <param name="dataView"></param>
        protected abstract void EvaluateModel(ITransformer model, IDataView dataView);

        private (ITransformer model, MLContext context) LoadModel()
        {
            if (!File.Exists(_modelPath))
                throw new ArgumentException("模型不存在");
            var context = new MLContext();
            var model = context.Model.Load(_modelPath, out _);
            return (model, context);
        }

        /// <summary> 模型评估 </summary>
        /// <param name="dataView"></param>
        public void Evaluate(IDataView dataView)
        {
            var (model, _) = LoadModel();
            EvaluateModel(model, dataView);
        }

        /// <summary> 模型预测 </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public TOutput Predict(TInput input)
        {
            var (model, context) = LoadModel();
            var engine = context.Model.CreatePredictionEngine<TInput, TOutput>(model);
            return engine.Predict(input);
        }

        /// <summary> 模型批量预测 </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public IEnumerable<TOutput> Predict(IEnumerable<TInput> inputs)
        {
            var (model, context) = LoadModel();
            var views = context.Data.LoadFromEnumerable(inputs);

            var predictions = model.Transform(views);
            var results = context.Data.CreateEnumerable<TOutput>(predictions, false);
            return results;
        }
    }
}
