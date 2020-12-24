using System;
using System.Threading.Tasks;
using ImageRecognizer;

namespace Sample
{
    class Program
    {
        static async Task Main(string[] args)
        {
            string imagesPath = (args.Length == 0) ? @"C:\Users\mary\source\repos3\ImageRecognizer\images" : args[0];



            ImageRecognizer.ImageRecognizer.onnxModelPath = (args.Length == 2) ? args[1] : @"C:\Users\mary\source\repos3\ImageRecognizer\ImageRecognizer\resnet34-v1-7.onnx";//"resnet34-v1-7.onnx";

            ImageRecognizer.ImageRecognizer.ImageRecognizerResultUpdate += OutputRecognitionHandler;

            await ImageRecognizer.ImageRecognizer.RecognitionAsync(imagesPath);
            
            

        }
        static void OutputRecognitionHandler(Prediction s)
        {
            Console.WriteLine($" {s.Label,15} {s.Confidence,15} ");
        }
    
    }
}
