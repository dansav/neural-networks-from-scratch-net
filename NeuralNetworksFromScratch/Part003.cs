using System;

namespace NeuralNetworksFromScratch
{
    public class Part003 : IPart
    {
        public void Run()
        {
            // https://www.youtube.com/watch?v=tMrbN67U9d4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=3
            Console.WriteLine("Part 3: The dot product");

            UseBasicProgrammingConstructs();

            UseDotProductSingleNeuron();

            UseDotProductLayerOfNeurons();
        }

        private void UseBasicProgrammingConstructs()
        {
            Console.WriteLine("-- A 'cleaner, more dynamic way'");

            var inputs = new[] { 1f, 2f, 3f, 2.5f };

            var weights = new[]
            {
                new[] { 0.2f, 0.8f, -0.5f, 1.0f } ,
                new[] { 0.5f, -0.91f, 0.26f, -0.5f },
                new[] { -0.26f, -0.27f, 0.17f, 0.87f }
            };

            var biases = new[] { 2f, 3f, 0.5f };

            const int neurons = 3;
            var layer_outputs = new float[neurons];
            for (int n = 0; n < neurons; n++)
            {
                var neuron_output = 0f;
                for (int i = 0; i < inputs.Length; i++)
                {
                    neuron_output += inputs[i] * weights[n][i];
                }
                neuron_output += biases[n];
                layer_outputs[n] = neuron_output;
            }

            Console.WriteLine($"output: {layer_outputs.Dump()}");
        }

        private void UseDotProductSingleNeuron()
        {
            Console.WriteLine("-- Use the dot product (simplified, single neuron)");

            var inputs = new[] { 1f, 2f, 3f, 2.5f };
            var weights = new[] { 0.2f, 0.8f, -0.5f, 1.0f };
            var bias = 2f;

            var output1 = inputs.Dot(weights) + bias;

            // With this data, produces the same outout
            var output2 = weights.Dot(inputs) + bias;

            Console.WriteLine($"output1: {output1:0.0}");
            Console.WriteLine($"output2: {output2:0.0}");
        }

        private void UseDotProductLayerOfNeurons()
        {
            Console.WriteLine("-- Use the dot product on a layer of neurons");

            var inputs = new[] { 1f, 2f, 3f, 2.5f };

            var weights = new[]
            {
                new[] { 0.2f, 0.8f, -0.5f, 1.0f } ,
                new[] { 0.5f, -0.91f, 0.26f, -0.5f },
                new[] { -0.26f, -0.27f, 0.17f, 0.87f }
            };

            var biases = new[] { 2f, 3f, 0.5f };

            // not possible here (much easier to see in C# than in Python).
            //var outputs1 = inputs.Dot(weights).Add(biases);
            var outputs2 = weights.Dot(inputs).Add(biases);
                       
            //Console.WriteLine($"outputs1: {outputs1.Dump()}");
            Console.WriteLine($"outputs2: {outputs2.Dump()}");
        }

    }
}
