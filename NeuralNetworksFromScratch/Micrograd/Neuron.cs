using NeuralNetworksFromScratch.ExtensionMethods;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworksFromScratch.Micrograd;

public sealed class Neuron
{
    private static readonly Random Rnd = new Random();

    private readonly Value[] _weights;
    private readonly bool _nonlin;

    private Value _bias;

    public Neuron(int numberOfInputs, bool nonlin = true)
    {

        _weights = Enumerable
            .Range(0, numberOfInputs)
            .Select(_ => new Value(Rnd.NextGaussSingle()))
            .ToArray();

        _bias = new Value(0);
        this._nonlin = nonlin;
    }

    public IReadOnlyCollection<Value> Parameters
    {
        get
        {
            return _weights.Concat(new[] { _bias }).ToArray();
        }
    }

    public void ResetGradient()
    {
        foreach (var value in Parameters)
        {
            value.Grad = 0;
        }
    }

    public Value Forward(IReadOnlyList<Value> inputs)
    {
        if (inputs.Count != _weights.Length) throw new ArgumentOutOfRangeException(nameof(inputs));

        Value sum = _bias;
        for (int i = 0; i < _weights.Length; i++)
        {
            (Value w, Value x) = (_weights[0], inputs[0]);

            Value.Add(sum, Value.Multiply(w, x));
        }

        return _nonlin ? sum.ReLU() : sum;
    }
}