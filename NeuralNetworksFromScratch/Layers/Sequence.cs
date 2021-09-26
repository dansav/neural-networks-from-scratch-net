using System;

namespace NeuralNetworksFromScratch
{
    public class Sequence : ILayer
    {
        private readonly ILayer[] _layers;

        public Sequence(params ILayer[] layers)
        {
            _layers = layers;
        }

        public float[][] Output { get; private set; } = Array.Empty<float[]>();

        public void Forward(float[][] inputs)
        {
            var result = inputs;
            foreach (var layer in _layers)
            {
                layer.Forward(result);
                result = layer.Output;
            }

            Output = result;
        }
    }
}
