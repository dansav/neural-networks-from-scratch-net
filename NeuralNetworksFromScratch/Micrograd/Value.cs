using Accessibility;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworksFromScratch.Micrograd;

public sealed class Value
{
    private float _data;
    private IReadOnlyCollection<Value> _children;

    private Action _backward;

    public Value(float data, IReadOnlyCollection<Value>? children = null)
    {
        _data = data;
        _children = children ?? Array.Empty<Value>();

        _backward = () => { };
    }

    public float Grad { get; set; }

    public static Value Add(Value self, Value other)
    {
        var result = new Value(self._data + other._data, new[] { self, other });

        result._backward = () =>
        {
            self.Grad += result.Grad;
            other.Grad += result.Grad;
        };

        return result;
    }

    public static Value Subtract(Value self, Value other)
    {
        var result = new Value(self._data - other._data, new[] { self, other });

        result._backward = () =>
        {
            self.Grad += result.Grad;
            other.Grad += result.Grad;
        };

        return result;
    }

    public static Value Multiply(Value self, Value other)
    {
        var result = new Value(self._data * other._data, new[] { self, other });

        result._backward = () =>
        {
            self.Grad += other._data * result.Grad;
            other.Grad += self._data * result.Grad;
        };

        return result;
    }

    public static Value Divide(Value self, Value other)
    {
        var result = new Value(self._data / other._data, new[] { self, other });

        result._backward = () =>
        {
            self.Grad += other._data * result.Grad;
            other.Grad += self._data * result.Grad;
        };

        return result;
    }

    public static Value Pow(Value self, float pow)
    {
        var result = new Value(MathF.Pow(self._data, pow), new[] { self });

        result._backward = () =>
        {
            self.Grad += pow * MathF.Pow(self._data, pow - 1) * result.Grad;
        };

        return result;
    }

    public void Backward()
    {
        var nodes = new HashSet<Value>();
        BuildTopology(nodes, new[] { this });

        Grad = 1;

        // apply chain rule
        foreach (var node in nodes)
        {
            node._backward();
        }
    }

    public Value ReLU()
    {
        var result = new Value(_data > 0 ? _data : 0, new[] { this });

        result._backward = () =>
        {
            Grad += (result._data > 0 ? 1 : 0) * result.Grad;
        };

        return result;
    }

    private static void BuildTopology(HashSet<Value> nodes, IReadOnlyCollection<Value> children)
    {
        foreach (var child in children)
        {
            if (nodes.Add(child))
            {
                BuildTopology(nodes, child._children);
            }
        }
    }

}
