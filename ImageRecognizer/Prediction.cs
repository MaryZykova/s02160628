using System;
using System.Collections.Generic;
using System.Text;

namespace ImageRecognizer
{
    public class Prediction
    {
        public string Label { get; set; }
        public float Confidence { get; set; }
        public string Path { get; set; }
    }
}
