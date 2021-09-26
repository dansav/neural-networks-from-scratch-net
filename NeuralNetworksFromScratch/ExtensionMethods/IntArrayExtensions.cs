using System;
using System.Diagnostics;
using System.Linq;

namespace NeuralNetworksFromScratch
{
    public static class IntArrayExtensions
    {
        public static string Dump(this int[] a)
        {
            return $"[ {string.Join(", ", a.Select(x => $"{x}")) } ]";
        }
    }
}