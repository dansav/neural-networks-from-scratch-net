using System.Linq;

namespace NeuralNetworksFromScratch
{
    public abstract class Loss
    {
        public float Calculate(float[][] output, int[] y) {
            var sampleLosses = Forward(output, y);

            // batch loss
            var dataLoss = sampleLosses.Average();
            return dataLoss;
        }

        protected abstract float[] Forward(float[][] prediction, int[] truth);
    }
}
