using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworksFromScratch.Micrograd;

public class MultiLayerPerceptron
{
    private IReadOnlyCollection<Layer> _layers;
    public MultiLayerPerceptron(params int[] layerSizes)
    {
        var layers = new List<Layer>();
        for (int i = 0; i < layerSizes.Length - 1; i++)
        {
            layers.Add(new Layer(layerSizes[i], layerSizes[i + 1], i + 1 == layerSizes.Length));
        }

        _layers = layers.ToArray();
    }

    public IReadOnlyCollection<Value> Parameters
    {
        get
        {
            return _layers.SelectMany(n => n.Parameters).ToArray();
        }
    }

    public void ResetGradient()
    {
        foreach (var value in Parameters)
        {
            value.Grad = 0;
        }
    }

    public IReadOnlyList<Value> Forward(IReadOnlyList<Value> inputs)
    {
        var x = inputs;
        foreach (var layer in _layers)
        {
            x = layer.Forward(x);
        }

        return x;
    }
}
