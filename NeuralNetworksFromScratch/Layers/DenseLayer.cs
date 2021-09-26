using System;
using System.Linq;

namespace NeuralNetworksFromScratch
{
    public class DenseLayer : ILayer
    {
        private static readonly Random rnd = new Random();

        private readonly int _inputs;
        private readonly int _neurons;

        private float[][] _weights;
        private float[] _biases;
        private float[][]? _output;

        public DenseLayer(int inputs, int neurons)
        {
            _inputs = inputs;
            _neurons = neurons;

            // Initialize weights with random values
            _weights = new float[inputs][];
            InitializeWeights(this, 0.10f);

            // Initialize biases
            // .NET already initializes arrays with 0-values.
            _biases = new float[neurons];
        }

        public float[][] Output => _output ?? new float[0][];

        public float[][] Weights
        {
            get {  return _weights; }
        }

        public float[] Biases
        {
            get { return _biases; }
        }

        public static void InitializeWeights(DenseLayer layer, float modifier = 1f)
        {
            for (int i = 0; i < layer._inputs; i++)
            {
                layer._weights[i] = new float[layer._neurons];
                for (int n = 0; n < layer._neurons; n++)
                {
                    layer._weights[i][n] = modifier * GetRandom();
                }
            }
        }

        public static void InitializeBiases(DenseLayer layer, float modifier = 1f)
        {
            for (int i = 0; i < layer._biases.Length; i++)
            {
                layer._biases[i] = modifier * GetRandom();
            }
        }

        public static void SetWeights(DenseLayer layer, float[][] weights)
        {
            for (int i = 0; i < layer._inputs; i++)
            {
                for (int n = 0; n < layer._neurons; n++)
                {
                    layer._weights[i][n] = weights[i][n];
                }
            }
        }

        public static void SetBiases(DenseLayer layer, float[] biases)
        {
            for (int i = 0; i < layer._biases.Length; i++)
            {
                layer._biases[i] = biases[i];
            }
        }

        public static void TweakWeights(DenseLayer layer, float modifier = 1f)
        {
            for (int i = 0; i < layer._inputs; i++)
            {
                for (int n = 0; n < layer._neurons; n++)
                {
                    layer._weights[i][n] += modifier * GetRandom();
                }
            }
        }

        public static void TweakBiases(DenseLayer layer, float modifier = 1f)
        {
            for (int i = 0; i < layer._biases.Length; i++)
            {
                layer._biases[i] += modifier * GetRandom();
            }
        }

        public void Forward(float[][] inputs)
        {
            _output = inputs.Dot(_weights).Add(_biases);
        }

        private static float GetRandom()
        {
            return 2f * (rnd.NextSingle() - 0.5f);
        }
    }
}
