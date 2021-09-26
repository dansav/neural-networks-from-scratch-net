using System;

namespace NeuralNetworksFromScratch
{
    public class Part001 : IPart
    {
        public void Run()
        {
            // https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3
            Console.WriteLine("Part 1: Intro and Neuron Code");

            // Coded with .NET 6.0 RC1
            var netRuntimeVersion = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription;
            Console.WriteLine($".NET Runtime Version: {netRuntimeVersion}");

            // What goes on in the forward pass inside a neuron
            var inputs = new[] { 1.2f, 5.1f, 2.1f };
            var weights = new[] { 3.1f, 2.1f, 8.7f };
            var bias = 3f;

            var output =
                inputs[0] * weights[0] +
                inputs[1] * weights[1] +
                inputs[2] * weights[2] +
                bias;

            Console.WriteLine($"output: {output:0.0}");
        }
    }
}
