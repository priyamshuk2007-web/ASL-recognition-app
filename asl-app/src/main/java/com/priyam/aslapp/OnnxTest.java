package com.priyam.aslapp;
import ai.onnxruntime.*;

import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;

public class OnnxTest {
    public static void main(String[] args) throws OrtException, IOException {
        System.out.println("Looking in: " + new java.io.File(".").getAbsolutePath());
        // Creates the environment ONNX Runtime needs to operate in — similar spirit
        // to OpenCV.loadLocally(), a one-time setup step.
        OrtEnvironment env = OrtEnvironment.getEnvironment();

     InputStream modelStream = OnnxTest.class.getClassLoader().getResourceAsStream("asl_letter_model.onnx");
byte[] modelBytes = modelStream.readAllBytes();
OrtSession session = env.createSession(modelBytes, new OrtSession.SessionOptions());
        // Build one fake input row: 63 numbers, all zeros for now — we're just testing
        // "does the pipe work," not testing real accuracy yet.
        float[] fakeLandmarks = {0.8715326189994812f,0.5290595293045044f,-2.534681584620557e-07f,0.836338996887207f,0.5726655721664429f,-0.22751586139202118f,0.7200502753257751f,0.615180492401123f,-0.3293963670730591f,0.5759848952293396f,0.6888835430145264f,-0.38154667615890503f,0.4390273094177246f,0.7406117916107178f,-0.4240446090698242f,0.6474884748458862f,0.30502617359161377f,-0.2462146133184433f,0.4142414629459381f,0.3562884032726288f,-0.3089042603969574f,0.2818739116191864f,0.4372629225254059f,-0.3574397563934326f,0.19281533360481262f,0.5126922130584717f,-0.38817059993743896f,0.6022912859916687f,0.32036733627319336f,-0.13671524822711945f,0.3839697241783142f,0.4074690341949463f,-0.20633961260318756f,0.42095497250556946f,0.4874863028526306f,-0.24586613476276398f,0.49834734201431274f,0.5323534607887268f,-0.2648605406284332f,0.559619665145874f,0.3570091724395752f,-0.03767136484384537f,0.3796044588088989f,0.43671369552612305f,-0.09229204058647156f,0.43216991424560547f,0.4982835054397583f,-0.11503202468156815f,0.5073196291923523f,0.528607189655304f,-0.1225023865699768f,0.5175097584724426f,0.3917105793952942f,0.04481396824121475f,0.38559240102767944f,0.434340238571167f,-0.00718059204518795f,0.406585156917572f,0.4800800681114197f,-0.030830305069684982f,0.45849165320396423f,0.5019078254699707f,-0.03821992874145508f};

        // ONNX Runtime expects a specific tensor shape: 1 row, 63 columns
        long[] shape = {1, 63};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(fakeLandmarks), shape);

        // Find out what your model's input is actually named
        String inputName = session.getInputNames().iterator().next();

        // Run it
        OrtSession.Result result = session.run(Collections.singletonMap(inputName, inputTensor));

        // Print whatever comes back
        String[] labels = (String[]) result.get(0).getValue();
        System.out.println("Predicted letter: " + labels[0]);
    }
}

