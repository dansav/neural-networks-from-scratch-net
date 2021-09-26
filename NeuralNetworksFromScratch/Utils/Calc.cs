using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworksFromScratch.Utils
{
    public static class Calc
    {
        public static float Accuracy(int[] predictions, int[] truths)
        {
            var matches = new float[truths.Length];
            for (int i = 0; i < truths.Length; i++)
            {
                matches[i] = truths[i] == predictions[i] ? 1.0f : 0.0f;
            }
            return matches.Average();
        }
    }
}
