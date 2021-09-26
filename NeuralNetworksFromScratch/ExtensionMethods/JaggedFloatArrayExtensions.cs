using System.Linq;

namespace NeuralNetworksFromScratch
{
    static class JaggedFloatArrayExtensions
    {
        public static float[][] Add(this float[][] a, float[] b)
        {
            var outputs = new float[a.Length][];
            for (int y = 0; y < a.Length; y++)
            {
                outputs[y] = a[y].Add(b);
            }
            return outputs;
        }

        public static float[] Dot(this float[][] weights, float[] inputs)
        {
            // Not (for now) optimized in any way for transparency
            var outputs = new float[weights.Length];
            for (var i = 0; i < weights.Length; i++)
            {
                outputs[i] = weights[i].Dot(inputs);
            }
            return outputs;
        }

        public static float[][] Dot(this float[][] inputs, float[][] weights)
        {
            // Not (for now) optimized in any way for transparency
            var outputs = new float[inputs.Length][];
            for (var y = 0; y < outputs.Length; y++)
            {
                outputs[y] = new float[weights[0].Length];
                for (var x = 0; x < weights[0].Length; x++)
                {
                    outputs[y][x] = inputs[y].Dot(Enumerable
                        .Range(0, weights.Length)
                        .Select(i => weights[i][x])
                        .ToArray());
                }
            }

            return outputs;
        }

        public static string Dump(this float[][] m)
        {
            return $"[\r\n  { string.Join("\r\n  ", m.Select(a => a.Dump())) }\r\n]";
        }

        public static float[][] Transpose(this float[][] a)
        {
            var output = new float[a[0].Length][];
            for (int x = 0; x < a[0].Length; x++)
            {
                output[x] = new float[a.Length];
                for (int y = 0; y < a.Length; y++)
                {
                    output[x][y] = a[y][x];
                }
            }

            return output;
        }
    }
}
