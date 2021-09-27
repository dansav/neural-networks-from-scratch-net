using NeuralNetworksFromScratch.Draw;
using NeuralNetworksFromScratch.Utils;
using System;
using System.Linq;

namespace NeuralNetworksFromScratch
{
    public class Part009 : IPart
    {
        public void Run()
        {
            // https://www.youtube.com/watch?v=txh3TQDwP1g&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=9
            Console.WriteLine("Part 9: Introducing Optimization and derivatives");

            TestVerticalData();

            TestSpiralData();

            RandomlySearchParametersForTheVerticalData();

            RandomlyTweakParametersForTheVerticalData();

            RandomlyTweakParametersForTheSpiralData();
        }

        private static void TestVerticalData()
        {
            Console.WriteLine("-- Using the vertical dataset.");

            var (X, y) = DataGenerator.GenerateVerticalData(100, 3);

            Show.Plot(Chart.CreateSeries(X, y), "Vertical data");
        }

        private static void TestSpiralData()
        {
            Console.WriteLine("-- Using the spiral dataset.");

            var (X, y) = DataGenerator.GenerateSpiralData(100, 3);

            Show.Plot(Chart.CreateSeries(X, y), "Spiral data");
        }

        private static void RandomlySearchParametersForTheVerticalData()
        {
            Console.WriteLine("-- Randomly search parameters for the vertical dataset.");

            var (X, y) = DataGenerator.GenerateVerticalData(100, 3);

            var dense1 = new DenseLayer(2, 3);
            var activation1 = new ReluActivation();
            var dense2 = new DenseLayer(3, 3);
            var activation2 = new SoftmaxActivation();

            var lossFunction = new CategoricalCrossEntropyLoss();

            var lowestLoss = 9_999_999f;
            var bestDense1Weights = dense1.Weights.Select(a => a.ToArray()).ToArray();
            var bestDense1Biases = dense1.Biases.ToArray();
            var bestDense2Weights = dense2.Weights.Select(a => a.ToArray()).ToArray();
            var bestDense2Biases = dense2.Biases.ToArray();

            for (int i = 0; i < 100_000; i++)
            {
                DenseLayer.Initialize(dense1, 0.05f);
                DenseLayer.Initialize(dense2, 0.05f);

                dense1.Forward(X);
                activation1.Forward(dense1.Output);
                dense2.Forward(activation1.Output);
                activation2.Forward(dense2.Output);

                var loss = lossFunction.Calculate(activation2.Output, y);
                var predictions = activation2.Output.ArgMax();
                var accuracy = Calc.Accuracy(predictions, y);

                if (loss < lowestLoss)
                {
                    Console.WriteLine($"New set of weights found, iteration: {i} - loss: {loss}, acc: {accuracy}");

                    lowestLoss = loss;
                    bestDense1Weights = dense1.Weights.Select(a => a.ToArray()).ToArray();
                    bestDense1Biases = dense1.Biases.ToArray();
                    bestDense2Weights = dense2.Weights.Select(a => a.ToArray()).ToArray();
                    bestDense2Biases = dense2.Biases.ToArray();
                }
            }
        }

        private static void RandomlyTweakParametersForTheVerticalData()
        {
            Console.WriteLine("-- Randomly tweak parameters for the vertical dataset.");

            var (X, y) = DataGenerator.GenerateVerticalData(100, 3);

            var dense1 = new DenseLayer(2, 3);
            var activation1 = new ReluActivation();
            var dense2 = new DenseLayer(3, 3);
            var activation2 = new SoftmaxActivation();

            var lossFunction = new CategoricalCrossEntropyLoss();

            var lowestLoss = 9_999_999f;
            var bestDense1Weights = dense1.Weights.Select(a => a.ToArray()).ToArray();
            var bestDense1Biases = dense1.Biases.ToArray();
            var bestDense2Weights = dense2.Weights.Select(a => a.ToArray()).ToArray();
            var bestDense2Biases = dense2.Biases.ToArray();

            for (int i = 0; i < 100_000; i++)
            {
                DenseLayer.Tweak(dense1, 0.05f);
                DenseLayer.Tweak(dense2, 0.05f);

                dense1.Forward(X);
                activation1.Forward(dense1.Output);
                dense2.Forward(activation1.Output);
                activation2.Forward(dense2.Output);

                var loss = lossFunction.Calculate(activation2.Output, y);
                var predictions = activation2.Output.ArgMax();
                var accuracy = Calc.Accuracy(predictions, y);

                if (loss < lowestLoss)
                {
                    Console.WriteLine($"New set of weights found, iteration: {i} - loss: {loss}, acc: {accuracy}");

                    lowestLoss = loss;
                    bestDense1Weights = dense1.Weights.Select(a => a.ToArray()).ToArray();
                    bestDense1Biases = dense1.Biases.ToArray();
                    bestDense2Weights = dense2.Weights.Select(a => a.ToArray()).ToArray();
                    bestDense2Biases = dense2.Biases.ToArray();
                }
                else
                {
                    DenseLayer.Set(dense1, bestDense1Weights, bestDense1Biases);
                    DenseLayer.Set(dense2, bestDense2Weights, bestDense2Biases);
                }
            }
        }

        private static void RandomlyTweakParametersForTheSpiralData()
        {
            Console.WriteLine("-- Randomly tweak parameters for the vertical dataset.");

            var (X, y) = DataGenerator.GenerateSpiralData(100, 3);

            var dense1 = new DenseLayer(2, 3);
            var activation1 = new ReluActivation();
            var dense2 = new DenseLayer(3, 3);
            var activation2 = new SoftmaxActivation();

            var lossFunction = new CategoricalCrossEntropyLoss();

            var lowestLoss = 9_999_999f;
            var bestDense1Weights = dense1.Weights.Select(a => a.ToArray()).ToArray();
            var bestDense1Biases = dense1.Biases.ToArray();
            var bestDense2Weights = dense2.Weights.Select(a => a.ToArray()).ToArray();
            var bestDense2Biases = dense2.Biases.ToArray();

            for (int i = 0; i < 100_000; i++)
            {
                DenseLayer.Tweak(dense1, 0.05f);
                DenseLayer.Tweak(dense2, 0.05f);

                dense1.Forward(X);
                activation1.Forward(dense1.Output);
                dense2.Forward(activation1.Output);
                activation2.Forward(dense2.Output);

                var loss = lossFunction.Calculate(activation2.Output, y);
                var predictions = activation2.Output.ArgMax();
                var accuracy = Calc.Accuracy(predictions, y);

                if (loss < lowestLoss)
                {
                    Console.WriteLine($"New set of weights found, iteration: {i} - loss: {loss}, acc: {accuracy}");

                    lowestLoss = loss;
                    bestDense1Weights = dense1.Weights.Select(a => a.ToArray()).ToArray();
                    bestDense1Biases = dense1.Biases.ToArray();
                    bestDense2Weights = dense2.Weights.Select(a => a.ToArray()).ToArray();
                    bestDense2Biases = dense2.Biases.ToArray();
                }
                else
                {
                    DenseLayer.Set(dense1, bestDense1Weights, bestDense1Biases);
                    DenseLayer.Set(dense2, bestDense2Weights, bestDense2Biases);
                }
            }
        }
    }
}
