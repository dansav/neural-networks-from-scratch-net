using System;
using System.Linq;

namespace NeuralNetworksFromScratch
{
    public class SoftmaxActivation : ILayer
    {
        // probabilities
        public float[][] Output { get; private set; } = Array.Empty<float[]>();

        public static float[] Softmax(float[] input)
        {
            var exp = Exponentiate(input);
            return Normalize(exp);
        }

        public void Forward(float[][] inputs)
        {
            Output = inputs
                .Select(Softmax)
                .ToArray();
        }

        private static float[] Exponentiate(float[] input)
        {
            var max = input.Max();
            return input
                //.Select(MathF.Exp) // Can overflow
                .Select(i => MathF.Exp(i - max))
                .ToArray();
        }

        private static float[] Normalize(float[] input)
        {
            var sum = input.Sum();
            return input
                .Select(i => i / sum)
                .ToArray();
        }
    }
}
