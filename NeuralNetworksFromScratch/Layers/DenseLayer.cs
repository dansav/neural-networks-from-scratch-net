using System;
using System.Linq;

namespace NeuralNetworksFromScratch
{
    public class DenseLayer : ILayer
    {
        private readonly float[][] _weights;
        
        private readonly float[] _biases;

        private float[][]? _output;

        public DenseLayer(int inputs, int neurons)
        {
            // Initialize weights with random values
            _weights = new float[inputs][];
            var r = new Random();
            for (int i = 0; i < inputs; i++)
            {
                _weights[i] = new float[neurons];
                for (int n = 0; n < neurons; n++)
                {
                    // standard normal distribution
                    _weights[i][n] = 0.2f * (r.NextSingle() - 0.5f);
                }
            }

            // Initialize biases
            // .NET already initializes arrays with 0-values.
            //_biases = Enumerable.Range(0, neurons).Select(i => 0.0f).ToArray();
            _biases = new float[neurons];
        }

        public float[][] Output => _output ?? new float[0][];

        public void Forward(float[][] inputs)
        {
            _output = inputs.Dot(_weights).Add(_biases);
        }
    }
}
