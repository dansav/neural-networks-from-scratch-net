using System;

namespace NeuralNetworksFromScratch.ExtensionMethods
{
    public static class RandomExtensions
    {
        public static float NextGaussSingle(this Random random)
        {
            float u1 = 1.0f - random.NextSingle();
            float u2 = 1.0f - random.NextSingle();
            return MathF.Sqrt(-2.0f * MathF.Log(u1))
                * MathF.Sin(2.0f * MathF.PI * u2);
        }
    }
}
