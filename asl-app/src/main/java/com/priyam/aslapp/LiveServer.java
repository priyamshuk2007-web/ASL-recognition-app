package com.priyam.aslapp;
import ai.onnxruntime.*;
import java.io.*;
import java.net.*;
import java.util.HashMap;
import java.util.Map;
import javax.swing.*;
import java.awt.*;

public class LiveServer {
    public static void main(String[] args) throws IOException, OrtException {
        StringBuilder sentence = new StringBuilder(); // put this OUTSIDE the while loop, near voteCounts
        JFrame frame = new JFrame("ASL Recognizer");
JLabel label = new JLabel("...", SwingConstants.CENTER);
label.setFont(new Font("Arial", Font.BOLD, 60));
frame.add(label);
frame.setSize(600, 200);
frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
frame.setVisible(true);
        int port = 5000;
        ProcessBuilder pb = new ProcessBuilder("py", "socket_test.py");
pb.directory(new File("../dataset"));
pb.redirectErrorStream(true);
Process pythonProcess = pb.start();

Map<String, Integer> voteCounts = new HashMap<>();
int MIN_VOTES = 10;        // a segment needs at least this many confident frames to count
int NO_HAND_LIMIT = 10;    // this many no-hand frames in a row = the hand has left
int noHandFrames = 0;      // running count of consecutive no-hand frames
double MIN_CONFIDENCE = 0.55; // frames below this are ignored
int MIN_BOTH_FRAMES = 8;   // two-hand frames in a row needed to count as "word finished"
int bothFrames = 0;        // running count of consecutive two-hand frames

// Print whatever Python outputs, so you can still see its errors/prints in Java's console
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
        // Opens the "phone line" and waits for someone to call in.
        ServerSocket serverSocket = new ServerSocket(port);
        System.out.println("Waiting for Python to connect on port " + port + "...");

        // This line pauses/blocks here until a connection actually arrives.
        Socket clientSocket = serverSocket.accept();
        System.out.println("Python connected!");

        // A reader to pull text, one line at a time, from that connection.
        BufferedReader reader = new BufferedReader(
            new InputStreamReader(clientSocket.getInputStream()));
            String line;
       // Set up the model once, before the loop — same as your OnnxTest code
OrtEnvironment env = OrtEnvironment.getEnvironment();
InputStream modelStream = LiveServer.class.getClassLoader().getResourceAsStream("asl_letter_model.onnx");
byte[] modelBytes = modelStream.readAllBytes();
OrtSession session = env.createSession(modelBytes, new OrtSession.SessionOptions());
String inputName = session.getInputNames().iterator().next();

while ((line = reader.readLine()) != null) {
    if (line.equals("BOTH")) {
        bothFrames++;
        noHandFrames = 0;
        if (bothFrames == MIN_BOTH_FRAMES) {
            commitSegment(voteCounts, sentence, MIN_VOTES);   // finish the pending letter first
            if (sentence.length() > 0 && sentence.charAt(sentence.length() - 1) != ' ') {
                sentence.append(' ');
            }
            updateDisplay(label, sentence);
        }
        continue;   // skip the model, the left hand must not create letters
    }
    bothFrames = 0;   // not a two-hand frame, reset the counter

    if (line.equals("NONE")) {
        noHandFrames++;
        if (noHandFrames == NO_HAND_LIMIT) {
            commitSegment(voteCounts, sentence, MIN_VOTES);
            updateDisplay(label, sentence);
        }
        continue;
    }
    // ...everything from "noHandFrames = 0;" onward stays exactly as it is
    noHandFrames = 0;   // a real hand frame arrived, so reset the gap counter
    String[] parts = line.split(",");

    float[] landmarks = new float[63];
    for (int i = 0; i < 63; i++) {
        landmarks[i] = Float.parseFloat(parts[i]);
    }

    long[] shape = {1, 63};
    OnnxTensor inputTensor = OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(landmarks), shape);
    OrtSession.Result result = session.run(java.util.Collections.singletonMap(inputName, inputTensor));

    String[] predictedLabel = (String[]) result.get(0).getValue();
    String predicted = predictedLabel[0];
    double confidence = 0;
Object probOut = result.get(1).getValue();
if (probOut instanceof java.util.List) {
    Object first = ((java.util.List<?>) probOut).get(0);
    Map<?, ?> probs = (first instanceof OnnxMap)
            ? (Map<?, ?>) ((OnnxMap) first).getValue()
            : (Map<?, ?>) first;
    confidence = ((Number) probs.get(predicted)).doubleValue();
} else {
    System.out.println("Probability output is: " + probOut.getClass());
}
if (confidence >= MIN_CONFIDENCE) {
    voteCounts.merge(predicted, 1, Integer::sum);
    System.out.println("COUNTED: " + predicted + "  " + confidence);

}
}


    

        System.out.println("Connection closed.");
        clientSocket.close();
        serverSocket.close();
}
    
    // Picks the winning letter of the current segment (if it has enough votes) and adds it.
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

// Shows only the newest part of the sentence that fits in the label.
static void updateDisplay(JLabel label, StringBuilder sentence) {
    String full = sentence.toString();
    FontMetrics fm = label.getFontMetrics(label.getFont());
    int available = label.getWidth() - 20;
    String shown = full;
    boolean trimmed = false;
    while (available > 0 && shown.length() > 1
            && fm.stringWidth((trimmed ? "..." : "") + shown) > available) {
        shown = shown.substring(1);   // drop the oldest character
        trimmed = true;
    }
    String text = trimmed ? "..." + shown : shown;
    SwingUtilities.invokeLater(() -> label.setText(text));
}

}