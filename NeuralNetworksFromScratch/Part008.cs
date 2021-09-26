using System;
using System.Linq;

namespace NeuralNetworksFromScratch
{
    public class Part008 : IPart
    {
        public void Run()
        {
            // https://www.youtube.com/watch?v=levekYbxauw&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=8
            Console.WriteLine("Part 8: Implementing Loss");

            ContinueExplainCrossEntropyOnBatch();

            ExplainArgMaxAndAccuracy();

            ApplyToModelFromPreviousParts();
        }

        private static void ContinueExplainCrossEntropyOnBatch()
        {
            Console.WriteLine("-- CrossEntropy on batch of samples");

            var softmaxOutputs = new[]
            {
               new[] { 0.7f, 0.1f, 0.2f } ,
               new[] { 0.1f, 0.5f, 0.4f },
               new []{ 0.02f, 0.9f, 0.08f }
           };

            var classTargets = new[] { 0, 1, 1 };

            var negLog = classTargets
                .Select((t, i) => -MathF.Log(softmaxOutputs[i][t]))
                .ToArray();

            Console.WriteLine($"loss (neg_log): {negLog.Dump()}");

            var averageLoss = negLog.Average();
            Console.WriteLine($"average_loss: {averageLoss}");

            // clip 1e-7
            Console.WriteLine($"Clip 0: {Clip(0)}, 1: {Clip(1)}");

            negLog = classTargets
                .Select((t, i) => -MathF.Log(Clip(softmaxOutputs[i][t])))
                .ToArray();

            Console.WriteLine($"loss (neg_log, clipped): {negLog.Dump()}");

            averageLoss = negLog.Average();
            Console.WriteLine($"average_loss: {averageLoss}");
        }

        private static float Clip(float value)
        {
            //return MathF.Min(1 - 1e-7f, MathF.Max(value, 1e-7f));
            return Math.Clamp(value, 1e-7f, 1f - 1e-7f);
        }

        private static void ExplainArgMaxAndAccuracy()
        {
            Console.WriteLine("-- ArgMax and accuracy");

            var softmaxOutputs = new[]
            {
               new[] { 0.7f, 0.2f, 0.1f } ,
               new[] { 0.5f, 0.1f, 0.4f },
               new []{ 0.02f, 0.9f, 0.08f }
           };

            var classTargets = new[] { 0, 1, 1 };
            var predictions = softmaxOutputs.Select(ArgMax).ToArray();

            Console.WriteLine($"Targets: {classTargets.Dump()}");
            Console.WriteLine($"Predictions: {predictions.Dump()}");

            Console.WriteLine($"Acc: {Acc(predictions, classTargets)}");
        }

        private static float Acc(int[] predictions, int[] truths)
        {
            var matches = new float[truths.Length];
            for (int i = 0; i < truths.Length; i++)
            {
                matches[i] = truths[i] == predictions[i] ? 1.0f : 0.0f;
            }
            return matches.Average();
        }

        private static int ArgMax(float[] values)
        {
            var (max, index) = values.Select((v, i) => (v, i)).Max();
            return index;
        }

        private static void ApplyToModelFromPreviousParts()
        {
            Console.WriteLine("-- Apply to model from previous parts.");

            var (X, y) = DataGenerator.GenerateSpiralData(100, 3);

            var model = new Sequence(
                new DenseLayer(2, 3),
                new ReluActivation(),
                new DenseLayer(3, 3),
                new SoftmaxActivation()
            );

            model.Forward(X);
            Console.WriteLine($"output (first 20): {model.Output[..20].Dump()}");

            var lossFunc = new CategoricalCrossEntropyLoss();

            var loss = lossFunc.Calculate(model.Output, y);

            Console.WriteLine($"Loss: {loss}");
        }
    }
}
