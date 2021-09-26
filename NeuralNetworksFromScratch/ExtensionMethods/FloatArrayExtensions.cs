using System;
using System.Diagnostics;
using System.Linq;

namespace NeuralNetworksFromScratch
{
    static class FloatArrayExtensions
    {
        public static string Dump(this float[] a)
        {
            return $"[ {string.Join(", ", a.Select(x => $"{x:0.0###}")) } ]";
        }
    }
}