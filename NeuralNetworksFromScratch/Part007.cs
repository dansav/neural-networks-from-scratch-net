using System;
using System.Linq;

namespace NeuralNetworksFromScratch
{
    public class Part007 : IPart
    {
        public void Run()
        {
            // https://www.youtube.com/watch?v=dEXPMQXoiLc&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=7
            Console.WriteLine("Part 7: Calculating Loss with Categorical Cross-Entropy");

            ExplainginLog();

            ExplainginOneHot();

            ExplainingCCE();
        }

        private static void ExplainginLog()
        {
            Console.WriteLine("-- Log and Exp");

            const float b = 5.2f;
            const float x = 1.6486586255873816f;

            Console.WriteLine($"b = {b}, log(b) = {MathF.Log(b)} (should equal x: {x})");

            Console.WriteLine($"x = {x}, exp(x) = {MathF.Exp(x)} (should equal b: {b})");

            Console.WriteLine($"x = {x}, exp(x) = {MathF.Exp(x)} (should equal pow(e, x): {MathF.Pow(MathF.E, x)})");
        }

        private static void ExplainginOneHot()
        {
            Console.WriteLine("-- One-hot");

            var oneHot = new[] { 0f, 0f, 1f };
            Console.WriteLine($"one-hot (hard coded): {oneHot.Dump()}");

            var classes = 3;
            var targetClass = 0; // The index of the "1"
            oneHot = Enumerable
                .Range(0, classes)
                .Select(i => i == targetClass ? 1f : 0f)
                .ToArray();

            Console.WriteLine($"one-hot (generated): {oneHot.Dump()}");
        }

        private static void ExplainingCCE()
        {
            Console.WriteLine("-- Categorical Cross-Entropy");

            var softmaxOutput = new[] { 0.7f, 0.1f, 0.2f };
            var targetOutput = new[] { 1.0f, 0f, 0f };

            var loss = -(
               targetOutput[0] * MathF.Log(softmaxOutput[0]) +
               targetOutput[1] * MathF.Log(softmaxOutput[1]) +
               targetOutput[2] * MathF.Log(softmaxOutput[2])
               );

            Console.WriteLine($"loss: {loss}");

            Console.WriteLine("-- Categorical Cross-Entropy 'optimized'");
            var targetClass = 0; // The index of the "1"

            loss = -MathF.Log(softmaxOutput[targetClass]);
            Console.WriteLine($"loss: {loss}");

            var confidence = new[] { 0.7f, 0.5f };
            foreach (var c in confidence)
            {
                Console.WriteLine($"confidence: {c} => loss: {-MathF.Log(c)}");
            }
        }
    }
}
