using NeuralNetworksFromScratch.ExtensionMethods;
using System;
using System.Linq;

namespace NeuralNetworksFromScratch
{
    public static class DataGenerator
    {
        static readonly Random _rnd = new();

        public static (float[][] X, int[] y) GenerateSpiralData(int points, int classes)
        {
            var X = Enumerable.Range(0, points * classes).Select(i => new float[] { 0.0f, 0.0f }).ToArray();
            var y = Enumerable.Range(0, points * classes).Select(i => 0).ToArray();

            for (int classNumber = 0; classNumber < classes; classNumber++)
            {
                var ix = Enumerable.Range(points * classNumber, points * (classNumber + 1) - points * classNumber);
                var r = LinSpace(0, 1, points); // radius
                var t = LinSpace(classNumber * 4, (classNumber + 1) * 4, points).Select(x => x + _rnd.NextGaussSingle() * 0.2f).ToArray();

                int i = 0;
                foreach (var x in ix)
                {
                    X[x][0] = r[i] * MathF.Sin(t[i] * 2.5f);
                    X[x][1] = r[i] * MathF.Cos(t[i] * 2.5f);
                    y[x] = classNumber;
                    i++;
                }
            }

            return (X, y);
        }

        public static (float[][] X, int[] y) GenerateVerticalData(int samples, int classes)
        {
            var X = Enumerable.Range(0, samples * classes).Select(i => new float[] { 0.0f, 0.0f }).ToArray();
            var y = Enumerable.Range(0, samples * classes).Select(i => 0).ToArray();

            for (int classNumber = 0; classNumber < classes; classNumber++)
            {
                var ix = Enumerable.Range(samples * classNumber, samples * (classNumber + 1) - samples * classNumber);
                foreach (var x in ix)
                {
                    X[x][0] = _rnd.NextGaussSingle() * 0.1f + classNumber / 3f;
                    X[x][1] = _rnd.NextGaussSingle() * 0.1f + 0.5f;
                    y[x] = classNumber;
                }
            }

            return (X, y);
        }

        private static float[] LinSpace(float start, float end, int count)
        {
            return Enumerable
                .Range(0, count)
                .Select(i => start + (end - start) * i / (count - 1f))
                .ToArray();
        }
    }
}
