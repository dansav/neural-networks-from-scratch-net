using NeuralNetworksFromScratch.Draw;
using System;

namespace NeuralNetworksFromScratch
{
    public class Part009 : IPart
    {
        public void Run()
        {
            // https://www.youtube.com/watch?v=txh3TQDwP1g&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=9
            Console.WriteLine("Part 9: Introducing Optimization and derivatives");

            TestVerticalData();

            TestSpiralData();
        }

        private void TestVerticalData()
        {
            Console.WriteLine("-- Using the vertical dataset.");

            var (X, y) = DataGenerator.GenerateVerticalData(100, 3);

            Show.Plot(Chart.CreateSeries(X, y), "Vertical data");
        }

        private void TestSpiralData()
        {
            Console.WriteLine("-- Using the spiral dataset.");

            var (X, y) = DataGenerator.GenerateSpiralData(100, 3);

            Show.Plot(Chart.CreateSeries(X, y), "Spiral data");
        }
    }
}
