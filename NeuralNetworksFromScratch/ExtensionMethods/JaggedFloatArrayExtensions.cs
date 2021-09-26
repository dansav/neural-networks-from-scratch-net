namespace NeuralNetworksFromScratch
{
    static class JaggedFloatArrayExtensions
    {
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
    }
}
