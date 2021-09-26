using System;
using System.Linq;

namespace NeuralNetworksFromScratch
{
    public class SigmoidActivation : ILayer
    {
        public float[][] Output { get; private set; } = new float[0][];

        public static float[] Sigmoid(float[] input)
        {
            //return input.Select(i => 1f / (1f + MathF.Exp(-i))).ToArray();

            // some alternatives from https://stackoverflow.com/questions/412019/math-optimization-in-c-sharp
            // I have not benchmarked...

            //return input.Select(i =>
            //{
            //    if (i < -45.0f) return 0.0f;
            //    if (i > 45.0f) return 1.0f;
            //    return 1.0f / (1.0f + MathF.Exp(-i));
            //}).ToArray();

            return input.Select(i => 0.5f / (0.5f + MathF.Tanh(i / 2f))).ToArray();

            //return input.Select(i => 0.5f + i / (2f + 2f * MathF.Abs(i))).ToArray();
        }

        public void Forward(float[][] inputs)
        {
            Output = inputs.Select(Sigmoid).ToArray();
        }
    }
}
