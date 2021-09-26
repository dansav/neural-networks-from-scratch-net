using System;
using System.Linq;

namespace NeuralNetworksFromScratch
{
    public class Part006 : IPart
    {
        public void Run()
        {
            // https://www.youtube.com/watch?v=omz_NdFgWyU&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=6
            Console.WriteLine("Part 6: Softmax activation");

            TestSoftMaxSimple();
            
            TestSoftMaxBatch();

            UseSoftmaxOnTheSpiralDataset();
        }

        private static void TestSoftMaxSimple()
        {
            Console.WriteLine("-- Test softmax simple");

            var inputs = new[] { 4.8f, 1.21f, 2.385f };

            // Exponentiate
            var exp_values = inputs.Select(MathF.Exp).ToArray();
            Console.WriteLine($"exp_values: {exp_values.Dump()}");

            // Normalize
            var norm_base = exp_values.Sum();
            var norm_values = exp_values.Select(v => v / norm_base).ToArray();
            Console.WriteLine($"norm_values: {norm_values.Dump()}");
            Console.WriteLine($"Norm_values sum (should be close to 1.0): {norm_values.Sum()}");
        }

        private static void TestSoftMaxBatch()
        {
            Console.WriteLine("-- Test softmax");

            var inputs = new[] {
                new [] { 4.8f, 1.21f, 2.385f },
                new [] { 8.9f, -1.81f, 0.2f },
                new [] { 1.41f, 1.051f, 0.026f },
            };

            var softmax = new SoftmaxActivation();

            softmax.Forward(inputs);
            Console.WriteLine($"outputs (Softmax): {softmax.Output.Dump()}");
        }

        private static void UseSoftmaxOnTheSpiralDataset()
        {
            Console.WriteLine("-- Use Softmax on the spiral dataset");

            const int classCount = 3;

            var (X, _) = DataGenerator.GenerateSpiralData(100, classCount);

            var model = new Sequence(
                new DenseLayer(2, 3),
                new ReluActivation(),
                new DenseLayer(3, classCount),
                new SoftmaxActivation()
            );

            model.Forward(X);

            Console.WriteLine($"outputs (first 20): {model.Output[..20].Dump()}");
        }
    }
}
