namespace NeuralNetworksFromScratch
{
    public interface ILayer
    {
        float[][] Output { get; }

        void Forward(float[][] inputs);
    }
}
