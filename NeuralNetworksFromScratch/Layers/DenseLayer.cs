using NeuralNetworksFromScratch.ExtensionMethods;
using System;

namespace NeuralNetworksFromScratch
{
    public class DenseLayer : ILayer
    {
        private static readonly Random _rnd = new();

        private readonly int _inputs;
        private readonly int _neurons;

        private readonly float[][] _weights;
        private readonly float[] _biases;

        private float[][]? _output;

        public DenseLayer(int inputs, int neurons)
        {
            _inputs = inputs;
            _neurons = neurons;

            _weights = new float[inputs][];
            _biases = new float[neurons];

            // Initialize weights and biases with random values
            Initialize(this, 0.10f);
        }

        public float[][] Output => _output ?? Array.Empty<float[]>();

        public float[][] Weights
        {
            get {  return _weights; }
        }

        public float[] Biases
        {
            get { return _biases; }
        }

        public static void Initialize(DenseLayer layer, float modifier = 1f)
        {
            for (int i = 0; i < layer._inputs; i++)
            {
                layer._weights[i] = new float[layer._neurons];
                for (int n = 0; n < layer._neurons; n++)
                {
                    layer._weights[i][n] = modifier * _rnd.NextGaussSingle();
                }
            }

            for (int i = 0; i < layer._biases.Length; i++)
            {
                layer._biases[i] = modifier * _rnd.NextGaussSingle();
            }
        }

        public static void Set(DenseLayer layer, float[][] weights, float[] biases)
        {
            for (int i = 0; i < layer._inputs; i++)
            {
                for (int n = 0; n < layer._neurons; n++)
                {
                    layer._weights[i][n] = weights[i][n];
                }
            }

            for (int i = 0; i < layer._biases.Length; i++)
            {
                layer._biases[i] = biases[i];
            }
        }

        public static void Tweak(DenseLayer layer, float modifier = 1f)
        {
            for (int i = 0; i < layer._inputs; i++)
            {
                for (int n = 0; n < layer._neurons; n++)
                {
                    layer._weights[i][n] += modifier * _rnd.NextGaussSingle();
                }
            }

            for (int i = 0; i < layer._biases.Length; i++)
            {
                layer._biases[i] += modifier * _rnd.NextGaussSingle();
            }
        }

        public void Forward(float[][] inputs)
        {
            _output = inputs.Dot(_weights).Add(_biases);
        }
    }
}
