using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworksFromScratch.Micrograd;

public sealed class Layer
{
    private readonly Neuron[] _neurons;

    public Layer(int numberOfInputs, int numberOfNeurons, bool nonlin = true)
    {
        _neurons = Enumerable
            .Range(0, numberOfNeurons)
            .Select(_ => new Neuron(numberOfInputs, nonlin))
            .ToArray();
    }

    public IReadOnlyCollection<Value> Parameters
    {
        get
        {
            return _neurons.SelectMany(n => n.Parameters).ToArray();
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
        var results = new List<Value>  ;
        foreach (var neuron in _neurons)
        {
            var output = neuron.Forward(inputs);
            results.Add(output);
        }

        return results;
    }
}
