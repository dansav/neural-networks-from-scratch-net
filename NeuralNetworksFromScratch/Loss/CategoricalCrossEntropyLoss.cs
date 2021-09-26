using System;
using System.Linq;

namespace NeuralNetworksFromScratch
{
    public class CategoricalCrossEntropyLoss : Loss
    {
        protected override float[] Forward(float[][] prediction, int[] truth)
        {
            var losses = truth
              .Select((t, i) => CalculateLoss(prediction[i][t]))
              .ToArray();

            return losses;
        }

        private static float CalculateLoss(float value)
        {
            return -MathF.Log(Math.Clamp(value, 1e-7f, 1f - 1e-7f));
        }
    }
}
