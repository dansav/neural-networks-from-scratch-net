using System;

namespace NeuralNetworksFromScratch
{
    public class Part005 : IPart
    {
        public void Run()
        {
            // https://www.youtube.com/watch?v=gmjzbpSVY1A&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=5
            Console.WriteLine("Part 5: Hidden Layer Activation Functions");

            WhatIsRelu();

            UseActivationLayer();

            UseActivationOnTheSpiralDataset();
        }

        private static void WhatIsRelu()
        {
            Console.WriteLine("-- What is ReLU");

            // ReLU
            var inputs = new[] { 0f, 2f, -1f, 3.3f, -2.7f, 1.1f, 2.2f, -100f };
            var outputs = new float[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                outputs[i] = inputs[i] > 0 ? inputs[i] : 0;
            }

            Console.WriteLine($"outputs: {outputs.Dump()}");
        }

        private static void UseActivationLayer()
        {
            Console.WriteLine("-- Using a ReLU activation layer");

            var X = new[]
            {
                new [] {1.0f, 2.0f, 3.0f, 2.5f },
                new [] {2.0f, 5.0f, -1.0f, 2.0f },
                new [] {-1.5f, 2.7f, 3.3f, -0.8f },
            };

            var layer1 = new DenseLayer(4, 5);
            var layer2 = new DenseLayer(5, 2);
            var layer3 = new ReluActivation();

            layer1.Forward(X);
            layer2.Forward(layer1.Output);
            layer3.Forward(layer2.Output);

            Console.WriteLine($"outputs (layer 3, ReLU): {layer3.Output.Dump()}");

            // Compare output with Sigmoid activation
            var sigmoid = new SigmoidActivation();
            sigmoid.Forward(layer2.Output);
            Console.WriteLine($"outputs (sigmoid): {sigmoid.Output.Dump()}");
        }

        private static void UseActivationOnTheSpiralDataset()
        {
            Console.WriteLine("-- Using a ReLU activation on the spiral dataset");
            
            var (X, y) = DataGenerator.GenerateSpiralData(100, 3);

            var layer1 = new DenseLayer(2, 5);
            layer1.Forward(X);
            Console.WriteLine($"outputs (layer 1): {layer1.Output[..5].Dump()}");

            var layer2 = new ReluActivation();
            layer2.Forward(layer1.Output);
            Console.WriteLine($"outputs (layer 2): {layer2.Output[..5].Dump()}");
        }
    }
}
