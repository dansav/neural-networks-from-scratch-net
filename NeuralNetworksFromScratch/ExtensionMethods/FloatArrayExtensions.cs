using System;
using System.Linq;

namespace NeuralNetworksFromScratch
{
    static class FloatArrayExtensions
    {
        public static float[] Add(this float[] a, float[] b)
        {
            var output = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                output[i] = a[i] + b[i];
            }
            return output;
        }

        public static float Dot(this float[] weights, float[] inputs)
        {
            if (weights.Length != inputs.Length) throw new ArgumentException("The length of inputs did not match the length of weights", nameof(inputs));

            // Not (for now) optimized in any way for transparency
            var output = 0.0f;
            for (var i = 0; i < weights.Length; i++)
            {
                output += weights[i] * inputs[i];
            }
            return output;
        }

        public static string Dump(this float[] a)
        {
            return $"[ {string.Join(", ", a.Select(x => $"{x:0.0###}")) } ]";
        }
    }
}