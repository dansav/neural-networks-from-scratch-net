using System.Linq;

namespace NeuralNetworksFromScratch
{
    public class ReluActivation : ILayer
    {
        public float[][] Output { get; private set; } = new float[0][];

        public static float[] ReLU(float[] input)
        {
            return input.Select(i => i > 0 ? i : 0).ToArray();
        }

        public void Forward(float[][] inputs)
        {
            Output = inputs.Select(ReLU).ToArray();
        }
    }
}
