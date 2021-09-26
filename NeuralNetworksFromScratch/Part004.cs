using System;

namespace NeuralNetworksFromScratch
{
    public class Part004 : IPart
    {
        public void Run()
        {
            // https://www.youtube.com/watch?v=TEWy9vZcxW4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=4
            Console.WriteLine("Part 4: Batches, Layers, and Objects");

            SingleSampleInputs();

            BatchInputs();

            AddAnotherLayer();

            DefineLayersUsingClasses();
        }

        private static void SingleSampleInputs()
        {
            Console.WriteLine("-- Single sample");

            var single_sample_inputs = new[] { 1f, 2f, 3f, 2.5f };

            var weights = new[]
            {
                new[] { 0.2f, 0.8f, -0.5f, 1.0f } ,
                new[] { 0.5f, -0.91f, 0.26f, -0.5f },
                new[] { -0.26f, -0.27f, 0.17f, 0.87f }
            };

            var biases = new[] { 2f, 3f, 0.5f };

            var outputs = weights.Dot(single_sample_inputs).Add(biases);

            Console.WriteLine($"outputs: {outputs.Dump()}");
        }

        private static void BatchInputs()
        {
            Console.WriteLine("-- Batch samples");

            var batch_inputs = new[]
            {
                new [] {1.0f, 2.0f, 3.0f, 2.5f },
                new [] {2.0f, 5.0f, -1.0f, 2.0f },
                new [] {-1.5f, 2.7f, 3.3f, -0.8f },
            };

            var weights = new[]
            {
                new[] { 0.2f, 0.8f, -0.5f, 1.0f } ,
                new[] { 0.5f, -0.91f, 0.26f, -0.5f },
                new[] { -0.26f, -0.27f, 0.17f, 0.87f }
            };
            weights = weights.Transpose();

            var biases = new[] { 2f, 3f, 0.5f };

            var outputs = batch_inputs.Dot(weights).Add(biases);

            Console.WriteLine($"outputs: {outputs.Dump()}");
        }

        private static void AddAnotherLayer()
        {
            Console.WriteLine("-- Add another layer");

            var batch_inputs = new[]
            {
                new [] {1.0f, 2.0f, 3.0f, 2.5f },
                new [] {2.0f, 5.0f, -1.0f, 2.0f },
                new [] {-1.5f, 2.7f, 3.3f, -0.8f },
            };

            // Define layer 1
            var weights1 = new[]
            {
                new[] { 0.2f, 0.8f, -0.5f, 1.0f } ,
                new[] { 0.5f, -0.91f, 0.26f, -0.5f },
                new[] { -0.26f, -0.27f, 0.17f, 0.87f }
            }.Transpose();
            var biases1 = new[] { 2f, 3f, 0.5f };

            // Define layer 2
            var weights2 = new[]
            {
                new[] { 0.1f, -0.14f, 0.5f } ,
                new[] { -0.5f, 0.12f, -0.33f },
                new[] { -0.44f, 0.73f, -0.13f }
            }.Transpose();
            var biases2 = new[] { -1f, 2f, -0.5f };

            // execute layer 1 (forward)
            var layer1_outputs = batch_inputs.Dot(weights1).Add(biases1);

            // execute layer 2 (forward)
            var layer2_outputs = layer1_outputs.Dot(weights2).Add(biases2);

            Console.WriteLine($"outputs (layer 2): {layer2_outputs.Dump()}");
        }

        private static void DefineLayersUsingClasses()
        {
            Console.WriteLine("-- Defining layers using classes");

            var X = new[]
            {
                new [] {1.0f, 2.0f, 3.0f, 2.5f },
                new [] {2.0f, 5.0f, -1.0f, 2.0f },
                new [] {-1.5f, 2.7f, 3.3f, -0.8f },
            };

            var layer1 = new DenseLayer(4, 5);
            var layer2 = new DenseLayer(5, 2);

            layer1.Forward(X);
            layer2.Forward(layer1.Output);

            Console.WriteLine($"outputs (layer 1): {layer1.Output.Dump()}");
            Console.WriteLine($"outputs (layer 2): {layer2.Output.Dump()}");
        }
    }
}
