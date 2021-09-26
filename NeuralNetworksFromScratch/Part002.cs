using System;

namespace NeuralNetworksFromScratch
{
    public class Part002 : IPart
    {
        public void Run()
        {
            // https://www.youtube.com/watch?v=lGLto9Xd7bU&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=2
            Console.WriteLine("Part 2: Coding a layer");

            RepeatPart1();

            AddAnInput();

            AddTwoNeurons();
        }

        private static void RepeatPart1()
        {
            Console.WriteLine($"-- {nameof(RepeatPart1)}");

            // Repeating what goes on in the forward pass inside a neuron
            // using values from the book.
            var inputs = new[] { 1f, 2f, 3f };
            var weights = new[] { 0.2f, 0.8f, -0.5f };
            var bias = 2f;

            var output =
                inputs[0] * weights[0] +
                inputs[1] * weights[1] +
                inputs[2] * weights[2] +
                bias;

            Console.WriteLine($"output: {output:0.0}");
        }

        private static void AddAnInput()
        {
            Console.WriteLine($"-- {nameof(AddAnInput)}");

            // add an input
            var inputs = new[] { 1f, 2f, 3f, 2.5f };
            var weights = new[] { 0.2f, 0.8f, -0.5f, 1.0f };
            var bias = 2f;

            var output =
                inputs[0] * weights[0] +
                inputs[1] * weights[1] +
                inputs[2] * weights[2] +
                inputs[3] * weights[3] +
                bias;

            Console.WriteLine($"output: {output:0.0}");
        }

        private static void AddTwoNeurons()
        {
            Console.WriteLine($"--- Add two neurons");

            var inputs = new[] { 1f, 2f, 3f, 2.5f};
            
            var weights1 = new[] { 0.2f, 0.8f, -0.5f, 1.0f };
            var weights2 = new[] { 0.5f, -0.91f, 0.26f, -0.5f };
            var weights3 = new[] { -0.26f, -0.27f, 0.17f, 0.87f };
            
            var bias1 = 2f;
            var bias2 = 3f;
            var bias3 = 0.5f;

            var output = new[]
            {
                inputs[0] * weights1[0] +
                inputs[1] * weights1[1] +
                inputs[2] * weights1[2] +
                inputs[3] * weights1[3] +
                bias1,
                inputs[0] * weights2[0] +
                inputs[1] * weights2[1] +
                inputs[2] * weights2[2] +
                inputs[3] * weights2[3] +
                bias2,
                inputs[0] * weights3[0] +
                inputs[1] * weights3[1] +
                inputs[2] * weights3[2] +
                inputs[3] * weights3[3] +
                bias3,
            };

            Console.WriteLine($"output: {output.Dump()}");
        }
    }
}
