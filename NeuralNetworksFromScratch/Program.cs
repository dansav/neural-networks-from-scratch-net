// See https://aka.ms/new-console-template for more information

using NeuralNetworksFromScratch;
using System;

// make sure we don't get localized number formatting
System.Threading.Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;

Console.WriteLine("Neural Networks from Scratch");

//new Part001().Run();
new Part002().Run();
