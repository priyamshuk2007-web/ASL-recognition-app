package com.priyam.aslapp;

import ai.onnxruntime.*;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;

public class OnnxTest {

    // Exact match to Python's normalize_landmarks() + extract_geometric_features()
    public static float[] buildFeatures77(float[] raw63) {
        float[] norm = new float[63];

        // 1. Subtract wrist (point 0)
        float wx = raw63[0], wy = raw63[1], wz = raw63[2];
        for (int i = 0; i < 21; i++) {
            norm[i * 3]     = raw63[i * 3]     - wx;
            norm[i * 3 + 1] = raw63[i * 3 + 1] - wy;
            norm[i * 3 + 2] = raw63[i * 3 + 2] - wz;
        }

        // 2. Divide by 3D distance to middle knuckle MCP (point 9)
        float scale = (float) Math.sqrt(
                norm[9 * 3] * norm[9 * 3] +
                norm[9 * 3 + 1] * norm[9 * 3 + 1] +
                norm[9 * 3 + 2] * norm[9 * 3 + 2]
        );
        if (scale < 1e-6f) scale = 1e-6f;
        for (int i = 0; i < 63; i++) {
            norm[i] /= scale;
        }

        // 3. Append 14 geometric distance features (63 -> 77)
        float[] features77 = new float[77];
        System.arraycopy(norm, 0, features77, 0, 63);
        int k = 63;

        // Thumb tip (4) to fingertips (8, 12, 16, 20)
        int[] tips = {8, 12, 16, 20};
        for (int idx : tips) features77[k++] = dist(norm, 4, idx);

        // Thumb tip (4) to middle knuckles / PIP joints (6, 10, 14, 18)
        int[] pips = {6, 10, 14, 18};
        for (int idx : pips) features77[k++] = dist(norm, 4, idx);

        // Adjacent fingertip spread (8-12, 12-16)
        features77[k++] = dist(norm, 8, 12);
        features77[k++] = dist(norm, 12, 16);

        // Finger curl ratios: dist(tip, 0) / (dist(mcp, 0) + 1e-6)
        int[][] tipMcpPairs = {{8, 5}, {12, 9}, {16, 13}, {20, 17}};
        for (int[] pair : tipMcpPairs) {
            features77[k++] = dist(norm, pair[0], 0) / (dist(norm, pair[1], 0) + 1e-6f);
        }

        return features77;
    }

    private static float dist(float[] pts, int i, int j) {
        float dx = pts[i * 3]     - pts[j * 3];
        float dy = pts[i * 3 + 1] - pts[j * 3 + 1];
        float dz = pts[i * 3 + 2] - pts[j * 3 + 2];
        return (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    public static void main(String[] args) throws OrtException, IOException {
        OrtEnvironment env = OrtEnvironment.getEnvironment();

        try (InputStream modelStream = OnnxTest.class.getClassLoader().getResourceAsStream("asl_letter_model.onnx")) {
            if (modelStream == null) {
                throw new RuntimeException("asl_letter_model.onnx not found in src/main/resources!");
            }
            byte[] modelBytes = modelStream.readAllBytes();

            try (OrtSession session = env.createSession(modelBytes, new OrtSession.SessionOptions())) {
                float[] rawFakeLandmarks = {
                    0.8715326189994812f,0.5290595293045044f,-2.534681584620557e-07f,
                    0.836338996887207f,0.5726655721664429f,-0.22751586139202118f,
                    0.7200502753257751f,0.615180492401123f,-0.3293963670730591f,
                    0.5759848952293396f,0.6888835430145264f,-0.38154667615890503f,
                    0.4390273094177246f,0.7406117916107178f,-0.4240446090698242f,
                    0.6474884748458862f,0.30502617359161377f,-0.2462146133184433f,
                    0.4142414629459381f,0.3562884032726288f,-0.3089042603969574f,
                    0.2818739116191864f,0.4372629225254059f,-0.3574397563934326f,
                    0.19281533360481262f,0.5126922130584717f,-0.38817059993743896f,
                    0.6022912859916687f,0.32036733627319336f,-0.13671524822711945f,
                    0.3839697241783142f,0.4074690341949463f,-0.20633961260318756f,
                    0.42095497250556946f,0.4874863028526306f,-0.24586613476276398f,
                    0.49834734201431274f,0.5323534607887268f,-0.2648605406284332f,
                    0.559619665145874f,0.3570091724395752f,-0.03767136484384537f,
                    0.3796044588088989f,0.43671369552612305f,-0.09229204058647156f,
                    0.43216991424560547f,0.4982835054397583f,-0.11503202468156815f,
                    0.5073196291923523f,0.528607189655304f,-0.1225023865699768f,
                    0.5175097584724426f,0.3917105793952942f,0.04481396824121475f,
                    0.38559240102767944f,0.434340238571167f,-0.00718059204518795f,
                    0.406585156917572f,0.4800800681114197f,-0.030830305069684982f,
                    0.45849165320396423f,0.5019078254699707f,-0.03821992874145508f
                };

                float[] input77 = buildFeatures77(rawFakeLandmarks);
                long[] shape = {1, 77};
                String inputName = session.getInputNames().iterator().next();

                try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input77), shape);
                     OrtSession.Result result = session.run(Collections.singletonMap(inputName, inputTensor))) {

                    String[] labels = (String[]) result.get(0).getValue();
                    float[][] probs = (float[][]) result.get(1).getValue();

                    float maxProb = 0f;
                    for (float p : probs[0]) if (p > maxProb) maxProb = p;

                    System.out.println("Predicted letter: " + labels[0] + " (confidence: " + String.format("%.2f%%", maxProb * 100) + ")");
                }
            }
        }
    }
}