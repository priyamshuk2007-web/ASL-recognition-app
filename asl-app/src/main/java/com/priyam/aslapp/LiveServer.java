package com.priyam.aslapp;

import ai.onnxruntime.*;
import java.io.*;
import java.net.*;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import javax.swing.*;
import java.awt.*;

public class LiveServer {

    // Class order matching model_export.py
    private static final String[] CLASSES = {
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"
    };

    public static void main(String[] args) throws IOException, OrtException {
        StringBuilder sentence = new StringBuilder();
        JFrame frame = new JFrame("ASL Recognizer");
        JLabel label = new JLabel("...", SwingConstants.CENTER);
        label.setFont(new Font("Arial", Font.BOLD, 60));
        frame.add(label);
        frame.setSize(600, 200);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);

        int port = 5000;
        ProcessBuilder pb = new ProcessBuilder("py", "socket_test.py");
        pb.directory(new File("dataset"));
        pb.redirectErrorStream(true);
        Process pythonProcess = pb.start();

        Map<String, Integer> voteCounts = new HashMap<>();
        int MIN_VOTES = 10;           // a segment needs at least this many confident frames to count
        int NO_HAND_LIMIT = 10;       // this many no-hand frames in a row = the hand has left
        int noHandFrames = 0;         // running count of consecutive no-hand frames
        double MIN_CONFIDENCE = 0.55; // frames below this are ignored
        int MIN_BOTH_FRAMES = 8;      // two-hand frames in a row needed to count as "word finished"
        int bothFrames = 0;           // running count of consecutive two-hand frames

        new Thread(() -> {
            try (BufferedReader pyReader = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()))) {
                String pyLine;
                while ((pyLine = pyReader.readLine()) != null) {
                    System.out.println("[Python] " + pyLine);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }).start();

        try (ServerSocket serverSocket = new ServerSocket(port);
             Socket clientSocket = serverSocket.accept();
             BufferedReader reader = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()))) {

            System.out.println("Waiting for Python to connect on port " + port + "...");
            System.out.println("Python connected!");
            String line;

            OrtEnvironment env = OrtEnvironment.getEnvironment();
            try (InputStream modelStream = LiveServer.class.getClassLoader().getResourceAsStream("asl_letter_model.onnx")) {
                if (modelStream == null) {
                    throw new FileNotFoundException("asl_letter_model.onnx not found in resources!");
                }
                byte[] modelBytes = modelStream.readAllBytes();

                try (OrtSession session = env.createSession(modelBytes, new OrtSession.SessionOptions())) {
                    String inputName = session.getInputNames().iterator().next();
                    long[] shape = {1, 77};

                    while ((line = reader.readLine()) != null) {
                        if (line.equals("BOTH")) {
                            bothFrames++;
                            noHandFrames = 0;
                            if (bothFrames == MIN_BOTH_FRAMES) {
                                commitSegment(voteCounts, sentence, MIN_VOTES);
                                if (sentence.length() > 0 && sentence.charAt(sentence.length() - 1) != ' ') {
                                    sentence.append(' ');
                                }
                                updateDisplay(label, sentence);
                            }
                            continue;
                        }
                        bothFrames = 0;

                        if (line.equals("NONE")) {
                            noHandFrames++;
                            if (noHandFrames == NO_HAND_LIMIT) {
                                commitSegment(voteCounts, sentence, MIN_VOTES);
                                updateDisplay(label, sentence);
                            }
                            continue;
                        }

                        noHandFrames = 0;
                        String[] parts = line.split(",");
                        if (parts.length < 63) continue;

                        float[] raw63 = new float[63];
                        for (int i = 0; i < 63; i++) {
                            raw63[i] = Float.parseFloat(parts[i]);
                        }

                        // Normalize (safe even if already normalized) + append 14 geometric distances -> 77 features
                        float[] features77 = buildFeatures77(raw63);

                        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(features77), shape);
                             OrtSession.Result result = session.run(Collections.singletonMap(inputName, inputTensor))) {

                            String[] predictedLabel = (String[]) result.get(0).getValue();
                            String predicted = predictedLabel[0];

                            double confidence = 0.0;
                            Object probOut = result.get(1).getValue();
                            if (probOut instanceof float[][]) {
                                float[] probs = ((float[][]) probOut)[0];
                                for (int i = 0; i < CLASSES.length && i < probs.length; i++) {
                                    if (CLASSES[i].equals(predicted)) {
                                        confidence = probs[i];
                                        break;
                                    }
                                }
                            } else if (probOut instanceof java.util.List) {
                                Object first = ((java.util.List<?>) probOut).get(0);
                                Map<?, ?> probs = (first instanceof OnnxMap)
                                        ? (Map<?, ?>) ((OnnxMap) first).getValue()
                                        : (Map<?, ?>) first;
                                confidence = ((Number) probs.get(predicted)).doubleValue();
                            }

                            if (confidence >= MIN_CONFIDENCE) {
                                voteCounts.merge(predicted, 1, Integer::sum);
                                System.out.println("COUNTED: " + predicted + "  " + String.format("%.3f", confidence));
                            }
                        }
                    }
                }
            }

            System.out.println("Connection closed.");
        }
    }

    /**
     * Ensures wrist-centered + palm-scaled normalization (idempotent if Python already normalized)
     * and appends the 14 geometric distance features used during MLP training.
     */
    static float[] buildFeatures77(float[] pts) {
        float[] norm = new float[63];
        float wx = pts[0], wy = pts[1], wz = pts[2];
        for (int i = 0; i < 21; i++) {
            norm[i * 3]     = pts[i * 3]     - wx;
            norm[i * 3 + 1] = pts[i * 3 + 1] - wy;
            norm[i * 3 + 2] = pts[i * 3 + 2] - wz;
        }

        float scale = (float) Math.sqrt(
                norm[9 * 3] * norm[9 * 3] +
                norm[9 * 3 + 1] * norm[9 * 3 + 1] +
                norm[9 * 3 + 2] * norm[9 * 3 + 2]
        );
        if (scale < 1e-6f) scale = 1e-6f;
        for (int i = 0; i < 63; i++) {
            norm[i] /= scale;
        }

        float[] out = new float[77];
        System.arraycopy(norm, 0, out, 0, 63);
        int k = 63;

        // 1. Thumb tip (4) to fingertips (8, 12, 16, 20)
        int[] tips = {8, 12, 16, 20};
        for (int idx : tips) out[k++] = dist(norm, 4, idx);

        // 2. Thumb tip (4) to PIP knuckles (6, 10, 14, 18)
        int[] pips = {6, 10, 14, 18};
        for (int idx : pips) out[k++] = dist(norm, 4, idx);

        // 3. Adjacent fingertip spread (8-12, 12-16)
        out[k++] = dist(norm, 8, 12);
        out[k++] = dist(norm, 12, 16);

        // 4. Finger curl ratios: dist(tip, 0) / (dist(mcp, 0) + 1e-6)
        int[][] tipMcpPairs = {{8, 5}, {12, 9}, {16, 13}, {20, 17}};
        for (int[] pair : tipMcpPairs) {
            out[k++] = dist(norm, pair[0], 0) / (dist(norm, pair[1], 0) + 1e-6f);
        }

        return out;
    }

    private static float dist(float[] pts, int i, int j) {
        float dx = pts[i * 3]     - pts[j * 3];
        float dy = pts[i * 3 + 1] - pts[j * 3 + 1];
        float dz = pts[i * 3 + 2] - pts[j * 3 + 2];
        return (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    static void commitSegment(Map<String, Integer> voteCounts, StringBuilder sentence, int minVotes) {
        String winner = null;
        int maxCount = -1;
        for (Map.Entry<String, Integer> entry : voteCounts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                winner = entry.getKey();
            }
        }
        if (winner != null && maxCount >= minVotes) {
            sentence.append(winner);
        }
        voteCounts.clear();
    }

    static void updateDisplay(JLabel label, StringBuilder sentence) {
        String full = sentence.toString();
        FontMetrics fm = label.getFontMetrics(label.getFont());
        int available = label.getWidth() - 20;
        String shown = full;
        boolean trimmed = false;
        while (available > 0 && shown.length() > 1
                && fm.stringWidth((trimmed ? "..." : "") + shown) > available) {
            shown = shown.substring(1);
            trimmed = true;
        }
        String text = trimmed ? "..." + shown : shown;
        SwingUtilities.invokeLater(() -> label.setText(text));
    }
}