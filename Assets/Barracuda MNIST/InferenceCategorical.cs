using System.Collections;
using System.Collections.Generic;
using System;
using System.Linq;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Rendering;
using TMPro;

// Based on https://www.youtube.com/watch?v=ggmArUbRvC4
public class InferenceCategorical : MonoBehaviour {
    public RenderTexture MNISTProjectorTexture; // This texture needs to be generated in MJPEGStreamDecoder.SendFrame()

    private Texture2D runtimeTexture;
    public NNModel modelAsset;

    [Header("Model")]
    private Model runtimeModel;
    public TextAsset labelAsset;
    private string[] labels;

    private IWorker worker;

    public TMP_Text predictionText;

    public bool updateAutomatically = true;
    public float updateFrequency = 0.1f;

    public int predictedValue;
    public float[] predictedArr;

    public TextureFormat renderTextureFormat = UnityEngine.TextureFormat.R8;
    public int channelCount = 1;


    void Start() {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(runtimeModel, WorkerFactory.Device.GPU);

        LoadLabels();

        if (updateAutomatically) {
            InvokeRepeating("Predict", updateFrequency, updateFrequency);
        }
    }

    void Update() {
        if (!updateAutomatically && Input.GetKeyDown(KeyCode.Space)) {
            Predict();
        }
    }

    void LoadLabels() {
        //get only items in quotes
        var stringArray = labelAsset.text.Split('"').Where((item, index) => index % 2 != 0);
        //get every other item
        labels = stringArray.Where((x, i) => i % 2 != 0).ToArray();
    }

    // Outputs an image based on a render texture. 
    public Texture2D RenderTexturetoTexture2D(RenderTexture rTex) {
        Texture2D dest = new Texture2D(rTex.width, rTex.height, renderTextureFormat, false);
        dest.Apply(false);
        Graphics.CopyTexture(rTex, dest);
        return dest;
    }
    public void SetPrediction(Tensor t, TMP_Text predictionText) {
        predictedArr = t.AsFloats();
        predictedValue = Array.IndexOf(predictedArr, predictedArr.Max());   //Argmax
        Debug.Log($" Predicted {predictedValue}");
        predictionText.SetText(labels[predictedValue]);
    }

    private void Predict() {
        runtimeTexture = RenderTexturetoTexture2D(MNISTProjectorTexture);

        // make a tensor out of a grayscale image
        var inputX = new Tensor(runtimeTexture, channelCount);

        Tensor outputY = worker.Execute(inputX).PeekOutput();
        SetPrediction(outputY, predictionText);

        //Manage Memory
        inputX.Dispose();
    }

    private void OnDestroy() {
        worker?.Dispose();
    }
}
