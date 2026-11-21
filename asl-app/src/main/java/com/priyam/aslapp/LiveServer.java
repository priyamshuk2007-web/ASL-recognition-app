package com.priyam.aslapp;

import ai.onnxruntime.*;
import java.awt.*;
import java.awt.datatransfer.StringSelection;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.*;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.border.LineBorder;

public class LiveServer {

    // Class order matching model_export.py
    private static final String[] CLASSES = {
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"
    };

    // UI & State components
    private static final StringBuilder sentence = new StringBuilder();
    private static JFrame frame;
    private static JLabel videoLabel;
    private static JLabel letterBadge;
    private static JProgressBar confBar;
    private static JLabel holdStatusLabel;
    private static JLabel handStatusPill;
    private static JTextArea sentenceArea;
    private static JLabel connectionStatusLabel;

    private static final AtomicReference<BufferedImage> latestFrame = new AtomicReference<>();
    private static final AtomicBoolean framePending = new AtomicBoolean(false);

    // Modern color palette
    private static final Color BG_DARK = new Color(18, 18, 22);
    private static final Color CARD_BG = new Color(28, 28, 34);
    private static final Color CARD_BORDER = new Color(50, 50, 62);
    private static final Color TEXT_PRIMARY = new Color(244, 244, 248);
    private static final Color TEXT_SECONDARY = new Color(160, 160, 175);
    private static final Color ACCENT_EMERALD = new Color(16, 185, 129);
    private static final Color ACCENT_SKY = new Color(14, 165, 233);
    private static final Color ACCENT_AMBER = new Color(245, 158, 11);

    public static synchronized void appendToSentence(String str) {
        sentence.append(str);
        updateDisplay();
    }

    public static synchronized void backspaceSentence() {
        if (sentence.length() > 0) {
            sentence.deleteCharAt(sentence.length() - 1);
            updateDisplay();
        }
    }

    public static synchronized void clearSentence() {
        sentence.setLength(0);
        updateDisplay();
    }

    private static void updateDisplay() {
        SwingUtilities.invokeLater(() -> {
            if (sentenceArea == null) return;
            String text;
            synchronized (sentence) {
                text = sentence.toString();
            }
            sentenceArea.setText(text.isEmpty() ? "Letters will appear here as you fingerspell..." : text);
            sentenceArea.setForeground(text.isEmpty() ? TEXT_SECONDARY : new Color(56, 189, 248));
            sentenceArea.setCaretPosition(sentenceArea.getDocument().getLength());
        });
    }

    private static void updateLiveMetrics(String letter, double confidence, String status) {
        SwingUtilities.invokeLater(() -> {
            if (letterBadge != null) {
                letterBadge.setText(letter);
                letterBadge.setForeground(letter.equals("-") ? TEXT_SECONDARY : ACCENT_EMERALD);
            }
            if (confBar != null) {
                int pct = (int) Math.round(confidence * 100);
                confBar.setValue(pct);
                confBar.setString(pct > 0 ? String.format("%.1f%% Confidence", confidence * 100) : "0.0% Confidence");
                if (pct >= 70) {
                    confBar.setForeground(ACCENT_EMERALD);
                } else if (pct >= 55) {
                    confBar.setForeground(ACCENT_AMBER);
                } else {
                    confBar.setForeground(TEXT_SECONDARY);
                }
            }
            if (holdStatusLabel != null) {
                holdStatusLabel.setText(status);
                if (status.startsWith("✓")) {
                    holdStatusLabel.setForeground(ACCENT_EMERALD);
                } else if (status.contains("Leading") || status.contains("Two hands")) {
                    holdStatusLabel.setForeground(ACCENT_AMBER);
                } else {
                    holdStatusLabel.setForeground(TEXT_SECONDARY);
                }
            }
        });
    }

    private static void updateHandStatus(String text, Color color) {
        SwingUtilities.invokeLater(() -> {
            if (handStatusPill != null) {
                handStatusPill.setText(text);
                handStatusPill.setForeground(color);
            }
        });
    }

    private static File findDatasetDirectory() {
        File dir = new File("dataset");
        if (dir.exists() && dir.isDirectory()) return dir;
        dir = new File("../dataset");
        if (dir.exists() && dir.isDirectory()) return dir;
        return new File(".");
    }

    private static String findPythonExecutable() {
        String[] candidates = {"py", "python3", "python"};
        for (String cmd : candidates) {
            try {
                Process p = new ProcessBuilder(cmd, "--version").start();
                if (p.waitFor() == 0) {
                    return cmd;
                }
            } catch (Exception ignored) {
            }
        }
        return "python";
    }

    private static JButton createStyledButton(String text, Color baseColor) {
        JButton btn = new JButton(text);
        btn.setFont(new Font("Segoe UI", Font.BOLD, 13));
        btn.setForeground(TEXT_PRIMARY);
        btn.setBackground(baseColor);
        btn.setFocusable(false);
        btn.setCursor(new Cursor(Cursor.HAND_CURSOR));
        btn.setBorder(BorderFactory.createCompoundBorder(
            new LineBorder(CARD_BORDER, 1, true),
            new EmptyBorder(8, 14, 8, 14)
        ));
        btn.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseEntered(MouseEvent e) {
                btn.setBackground(baseColor.brighter());
            }
            @Override
            public void mouseExited(MouseEvent e) {
                btn.setBackground(baseColor);
            }
        });
        return btn;
    }

    private static JPanel createCard(String title) {
        JPanel card = new JPanel(new BorderLayout());
        card.setBackground(CARD_BG);
        card.setBorder(BorderFactory.createCompoundBorder(
            new LineBorder(CARD_BORDER, 1, true),
            new EmptyBorder(12, 14, 12, 14)
        ));

        if (title != null && !title.isEmpty()) {
            JLabel titleLabel = new JLabel(title);
            titleLabel.setFont(new Font("Segoe UI", Font.BOLD, 14));
            titleLabel.setForeground(TEXT_SECONDARY);
            titleLabel.setBorder(new EmptyBorder(0, 0, 8, 0));
            card.add(titleLabel, BorderLayout.NORTH);
        }
        return card;
    }

    private static void setupUI(Process pythonProcess) {
        try {
            UIManager.setLookAndFeel(UIManager.getCrossPlatformLookAndFeelClassName());
        } catch (Exception ignored) {
        }

        frame = new JFrame("ASL Recognition Studio");
        frame.setLayout(new BorderLayout(14, 14));
        frame.getContentPane().setBackground(BG_DARK);
        ((JPanel) frame.getContentPane()).setBorder(new EmptyBorder(14, 16, 14, 16));

        // 1. TOP HEADER BAR
        JPanel headerPanel = new JPanel(new BorderLayout());
        headerPanel.setBackground(BG_DARK);
        headerPanel.setBorder(new EmptyBorder(0, 4, 6, 4));

        JLabel appTitle = new JLabel("✋  ASL Fingerspelling Recognizer");
        appTitle.setFont(new Font("Segoe UI", Font.BOLD, 22));
        appTitle.setForeground(TEXT_PRIMARY);

        connectionStatusLabel = new JLabel("● Streaming Active (Port 5000)");
        connectionStatusLabel.setFont(new Font("Segoe UI", Font.BOLD, 13));
        connectionStatusLabel.setForeground(ACCENT_EMERALD);

        headerPanel.add(appTitle, BorderLayout.WEST);
        headerPanel.add(connectionStatusLabel, BorderLayout.EAST);
        frame.add(headerPanel, BorderLayout.NORTH);

        // 2. MAIN SPLIT CONTENT
        JPanel contentPanel = new JPanel(new BorderLayout(16, 0));
        contentPanel.setBackground(BG_DARK);

        // LEFT COLUMN: CAMERA FEED CARD
        JPanel cameraCard = createCard("LIVE CAMERA & LANDMARK TRACKING");
        cameraCard.setPreferredSize(new Dimension(670, 560));

        videoLabel = new JLabel("Connecting to Camera Feed...", SwingConstants.CENTER);
        videoLabel.setPreferredSize(new Dimension(640, 480));
        videoLabel.setBackground(Color.BLACK);
        videoLabel.setOpaque(true);
        videoLabel.setForeground(TEXT_SECONDARY);
        videoLabel.setFont(new Font("Segoe UI", Font.PLAIN, 15));
        cameraCard.add(videoLabel, BorderLayout.CENTER);

        // Sub-bar under video
        JPanel videoFooter = new JPanel(new BorderLayout());
        videoFooter.setBackground(CARD_BG);
        videoFooter.setBorder(new EmptyBorder(10, 4, 2, 4));

        handStatusPill = new JLabel("⚪ Searching for Hand...", SwingConstants.LEFT);
        handStatusPill.setFont(new Font("Segoe UI", Font.BOLD, 13));
        handStatusPill.setForeground(TEXT_SECONDARY);

        JLabel resLabel = new JLabel("640 × 480  •  MediaPipe HandLandmarker", SwingConstants.RIGHT);
        resLabel.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        resLabel.setForeground(TEXT_SECONDARY);

        videoFooter.add(handStatusPill, BorderLayout.WEST);
        videoFooter.add(resLabel, BorderLayout.EAST);
        cameraCard.add(videoFooter, BorderLayout.SOUTH);

        contentPanel.add(cameraCard, BorderLayout.WEST);

        // RIGHT COLUMN: DASHBOARD & CONTROLS
        JPanel rightPanel = new JPanel();
        rightPanel.setLayout(new BoxLayout(rightPanel, BoxLayout.Y_AXIS));
        rightPanel.setBackground(BG_DARK);
        rightPanel.setPreferredSize(new Dimension(420, 560));

        // CARD A: LIVE RECOGNITION
        JPanel liveCard = createCard("REAL-TIME PREDICTION");
        liveCard.setLayout(new BorderLayout(12, 8));
        liveCard.setMaximumSize(new Dimension(450, 160));

        // Big Letter Badge
        letterBadge = new JLabel("-", SwingConstants.CENTER);
        letterBadge.setFont(new Font("Segoe UI", Font.BOLD, 68));
        letterBadge.setForeground(TEXT_SECONDARY);
        letterBadge.setPreferredSize(new Dimension(100, 100));
        letterBadge.setBackground(new Color(20, 20, 26));
        letterBadge.setOpaque(true);
        letterBadge.setBorder(new LineBorder(CARD_BORDER, 1, true));

        // Confidence & Hold stats
        JPanel metricsPanel = new JPanel();
        metricsPanel.setLayout(new BoxLayout(metricsPanel, BoxLayout.Y_AXIS));
        metricsPanel.setBackground(CARD_BG);
        metricsPanel.setBorder(new EmptyBorder(4, 8, 4, 4));

        confBar = new JProgressBar(0, 100);
        confBar.setValue(0);
        confBar.setStringPainted(true);
        confBar.setString("0.0% Confidence");
        confBar.setFont(new Font("Segoe UI", Font.BOLD, 13));
        confBar.setBackground(new Color(40, 40, 50));
        confBar.setForeground(ACCENT_EMERALD);
        confBar.setBorder(new LineBorder(CARD_BORDER, 1, true));
        confBar.setPreferredSize(new Dimension(220, 26));
        confBar.setMaximumSize(new Dimension(320, 26));

        holdStatusLabel = new JLabel("Show hand to spell • Remove hand to confirm");
        holdStatusLabel.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        holdStatusLabel.setForeground(TEXT_SECONDARY);
        holdStatusLabel.setBorder(new EmptyBorder(10, 0, 0, 0));

        metricsPanel.add(confBar);
        metricsPanel.add(holdStatusLabel);

        liveCard.add(letterBadge, BorderLayout.WEST);
        liveCard.add(metricsPanel, BorderLayout.CENTER);

        // CARD B: FORMED TEXT & SENTENCE
        JPanel textCard = createCard("RECOGNIZED SENTENCE");
        textCard.setLayout(new BorderLayout(8, 10));

        sentenceArea = new JTextArea("Letters will appear here as you fingerspell...");
        sentenceArea.setFont(new Font("Segoe UI", Font.BOLD, 22));
        sentenceArea.setBackground(new Color(20, 20, 26));
        sentenceArea.setForeground(TEXT_SECONDARY);
        sentenceArea.setLineWrap(true);
        sentenceArea.setWrapStyleWord(true);
        sentenceArea.setEditable(false);
        sentenceArea.setBorder(new EmptyBorder(10, 12, 10, 12));

        JScrollPane scrollPane = new JScrollPane(sentenceArea);
        scrollPane.setBorder(new LineBorder(CARD_BORDER, 1, true));
        scrollPane.setBackground(new Color(20, 20, 26));
        scrollPane.setPreferredSize(new Dimension(380, 140));
        textCard.add(scrollPane, BorderLayout.CENTER);

        // Action Buttons: Del, Clear, Copy (Space is exclusively done via two hands)
        JPanel buttonRow = new JPanel(new GridLayout(1, 3, 10, 0));
        buttonRow.setBackground(CARD_BG);

        JButton backspaceBtn = createStyledButton("⌫ Del", new Color(45, 45, 55));
        JButton clearBtn = createStyledButton("🗑 Clear", new Color(55, 35, 40));
        JButton copyBtn = createStyledButton("📋 Copy", new Color(30, 50, 45));

        backspaceBtn.addActionListener(e -> backspaceSentence());
        clearBtn.addActionListener(e -> clearSentence());
        copyBtn.addActionListener(e -> {
            String text;
            synchronized (sentence) {
                text = sentence.toString();
            }
            if (!text.isEmpty()) {
                Toolkit.getDefaultToolkit().getSystemClipboard().setContents(new StringSelection(text), null);
                copyBtn.setText("✓ Copied!");
                Timer t = new Timer(1500, evt -> copyBtn.setText("📋 Copy"));
                t.setRepeats(false);
                t.start();
            }
        });

        buttonRow.add(backspaceBtn);
        buttonRow.add(clearBtn);
        buttonRow.add(copyBtn);
        textCard.add(buttonRow, BorderLayout.SOUTH);

        // CARD C: GESTURE & SHORTCUT CHEAT SHEET
        JPanel guideCard = createCard("GESTURES & CONTROLS");
        guideCard.setMaximumSize(new Dimension(450, 150));
        JPanel guideContent = new JPanel(new GridLayout(4, 1, 4, 4));
        guideContent.setBackground(CARD_BG);

        guideContent.add(createGuideRow("🖐  Confirm Letter", "Hold sign pose, then remove hand from camera"));
        guideContent.add(createGuideRow("🙌  Add Space", "Show both hands simultaneously"));
        guideContent.add(createGuideRow("🔄  Hand Support", "Works with either Left or Right hand"));
        guideContent.add(createGuideRow("⌨  Hotkeys", "[Backspace] Delete  |  [Esc] Clear"));
        guideCard.add(guideContent, BorderLayout.CENTER);

        rightPanel.add(liveCard);
        rightPanel.add(Box.createVerticalStrut(12));
        rightPanel.add(textCard);
        rightPanel.add(Box.createVerticalStrut(12));
        rightPanel.add(guideCard);

        contentPanel.add(rightPanel, BorderLayout.CENTER);
        frame.add(contentPanel, BorderLayout.CENTER);

        // Global Key Bindings: Backspace and Escape (Space is strictly two-hand gesture)
        JRootPane rootPane = frame.getRootPane();
        rootPane.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW).put(KeyStroke.getKeyStroke(KeyEvent.VK_BACK_SPACE, 0), "backspace");
        rootPane.getActionMap().put("backspace", new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                backspaceSentence();
            }
        });

        rootPane.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW).put(KeyStroke.getKeyStroke(KeyEvent.VK_ESCAPE, 0), "clear");
        rootPane.getActionMap().put("clear", new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                clearSentence();
            }
        });

        frame.setSize(1120, 680);
        frame.setMinimumSize(new Dimension(1080, 640));
        frame.setLocationRelativeTo(null);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        frame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                if (pythonProcess != null && pythonProcess.isAlive()) {
                    pythonProcess.destroyForcibly();
                }
                System.exit(0);
            }
        });

        frame.setVisible(true);
    }

    private static JPanel createGuideRow(String title, String desc) {
        JPanel row = new JPanel(new BorderLayout());
        row.setBackground(CARD_BG);
        JLabel t = new JLabel(title);
        t.setFont(new Font("Segoe UI", Font.BOLD, 12));
        t.setForeground(TEXT_PRIMARY);
        JLabel d = new JLabel(desc);
        d.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        d.setForeground(TEXT_SECONDARY);
        row.add(t, BorderLayout.WEST);
        row.add(d, BorderLayout.EAST);
        return row;
    }

    public static void main(String[] args) throws IOException, OrtException {
        int port = 5000;
        File datasetDir = findDatasetDirectory();

        // 1. Bind ServerSocket first to eliminate the startup race condition
        try (ServerSocket serverSocket = new ServerSocket(port)) {
            serverSocket.setSoTimeout(30000); // 30-second timeout for Python to start & connect

            // 2. Discover Python executable and start subprocess
            String pythonCmd = findPythonExecutable();
            System.out.println("Using Python launcher: " + pythonCmd);
            System.out.println("Running from dataset directory: " + datasetDir.getAbsolutePath());

            ProcessBuilder pb = new ProcessBuilder(pythonCmd, "socket_test.py");
            pb.directory(datasetDir);
            pb.redirectErrorStream(true);
            Process pythonProcess = pb.start();

            // 3. Register JVM shutdown hook to kill orphaned process
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                if (pythonProcess != null && pythonProcess.isAlive()) {
                    pythonProcess.destroyForcibly();
                }
            }));

            // 4. Initialize Modern Swing UI on Event Dispatch Thread
            SwingUtilities.invokeLater(() -> setupUI(pythonProcess));

            // 5. Pipe Python output to Java console
            new Thread(() -> {
                try (BufferedReader pyReader = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()))) {
                    String pyLine;
                    while ((pyLine = pyReader.readLine()) != null) {
                        System.out.println("[Python] " + pyLine);
                    }
                } catch (IOException ignored) {
                }
            }).start();

            System.out.println("Waiting for Python to connect on port " + port + "...");

            try (Socket clientSocket = serverSocket.accept();
                 DataInputStream dis = new DataInputStream(new BufferedInputStream(clientSocket.getInputStream(), 65536));
                 OrtEnvironment env = OrtEnvironment.getEnvironment();
                 OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions()) {

                System.out.println("Python connected!");

                try (InputStream modelStream = LiveServer.class.getClassLoader().getResourceAsStream("asl_letter_model.onnx")) {
                    if (modelStream == null) {
                        throw new FileNotFoundException("asl_letter_model.onnx not found in resources!");
                    }
                    byte[] modelBytes = modelStream.readAllBytes();

                    try (OrtSession session = env.createSession(modelBytes, sessionOptions)) {
                        String inputName = session.getInputNames().iterator().next();
                        long[] shape = {1, 77};

                        Map<String, Integer> voteCounts = new HashMap<>();
                        int MIN_VOTES = 10;           // votes required in a segment to confirm a letter
                        int NO_HAND_LIMIT = 10;       // consecutive no-hand frames = hand has left, confirming letter
                        int noHandFrames = 0;
                        double MIN_CONFIDENCE = 0.55; // frames below this are ignored
                        int MIN_BOTH_FRAMES = 8;      // two-hand frames in a row needed to insert space
                        int bothFrames = 0;
                        int postBothCooldown = 0;

                        String currentLeader = null;
                        int currentLeaderVotes = 0;

                        while (true) {
                            int metaLen;
                            try {
                                metaLen = dis.readInt();
                            } catch (EOFException e) {
                                break;
                            }
                            int jpegLen = dis.readInt();

                            byte[] metaBytes = new byte[metaLen];
                            dis.readFully(metaBytes);

                            byte[] jpegBytes = new byte[jpegLen];
                            dis.readFully(jpegBytes);

                            String metaStr = new String(metaBytes, StandardCharsets.UTF_8);

                            // Update live video feed on Swing UI
                            BufferedImage img = ImageIO.read(new ByteArrayInputStream(jpegBytes));
                            if (img != null) {
                                latestFrame.set(img);
                                if (framePending.compareAndSet(false, true)) {
                                    SwingUtilities.invokeLater(() -> {
                                        framePending.set(false);
                                        BufferedImage f = latestFrame.get();
                                        if (f != null && videoLabel != null) {
                                            videoLabel.setIcon(new ImageIcon(f));
                                            videoLabel.setText("");
                                        }
                                    });
                                }
                            }

                            // 1. Two-Hand Space Gesture
                            if (metaStr.equals("BOTH")) {
                                bothFrames++;
                                noHandFrames = 0;
                                voteCounts.clear();
                                currentLeader = null;
                                currentLeaderVotes = 0;
                                postBothCooldown = 12;

                                updateHandStatus("🟡 Both Hands (SPACE)", ACCENT_AMBER);
                                updateLiveMetrics("␣", 1.0, "Two hands detected — Space mode");

                                if (bothFrames == MIN_BOTH_FRAMES) {
                                    synchronized (sentence) {
                                        if (sentence.length() > 0 && sentence.charAt(sentence.length() - 1) != ' ') {
                                            sentence.append(' ');
                                        }
                                    }
                                    updateDisplay();
                                    updateLiveMetrics("␣", 1.0, "✓ Space inserted!");
                                }
                                continue;
                            }
                            bothFrames = 0;

                            // 2. Hand Removed (Original Design: remove hand to confirm letter)
                            if (metaStr.equals("NONE")) {
                                noHandFrames++;
                                updateHandStatus("⚪ Searching for Hand...", TEXT_SECONDARY);

                                if (noHandFrames == NO_HAND_LIMIT) {
                                    if (postBothCooldown <= 0) {
                                        String confirmed = commitSegment(voteCounts, MIN_VOTES);
                                        if (confirmed != null) {
                                            updateLiveMetrics(confirmed, 1.0, "✓ Confirmed letter: " + confirmed);
                                        } else {
                                            updateLiveMetrics("-", 0.0, "Hand removed (not enough votes)");
                                        }
                                    }
                                    voteCounts.clear();
                                    currentLeader = null;
                                    currentLeaderVotes = 0;
                                } else if (noHandFrames > NO_HAND_LIMIT && noHandFrames % 15 == 0) {
                                    updateLiveMetrics("-", 0.0, "Show hand to spell • Remove hand to confirm");
                                }
                                continue;
                            }

                            // 3. Hand Present: Accumulate Votes while Hand Remains in Frame
                            noHandFrames = 0;
                            if (postBothCooldown > 0) {
                                postBothCooldown--;
                                continue;
                            }

                            if (metaStr.startsWith("HAND:")) {
                                String[] sections = metaStr.split(":", 3);
                                if (sections.length < 3) continue;

                                String handLabel = sections[1];
                                String csv = sections[2];

                                if ("Left".equalsIgnoreCase(handLabel)) {
                                    updateHandStatus("🔵 Left Hand (Mirrored)", ACCENT_SKY);
                                } else {
                                    updateHandStatus("🟢 Right Hand Active", ACCENT_EMERALD);
                                }

                                String[] parts = csv.split(",");
                                if (parts.length < 63) continue;

                                float[] raw63 = new float[63];
                                for (int i = 0; i < 63; i++) {
                                    raw63[i] = Float.parseFloat(parts[i]);
                                }

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
                                        Object val = probs.get(predicted);
                                        if (val instanceof Number) {
                                            confidence = ((Number) val).doubleValue();
                                        }
                                    }

                                    // Accumulate votes while hand is in frame
                                    if (confidence >= MIN_CONFIDENCE) {
                                        int count = voteCounts.merge(predicted, 1, Integer::sum);
                                        if (count > currentLeaderVotes) {
                                            currentLeader = predicted;
                                            currentLeaderVotes = count;
                                        }
                                        String status = String.format("Leading: %s (%d frames) • Remove hand to confirm", currentLeader, currentLeaderVotes);
                                        updateLiveMetrics(predicted, confidence, status);
                                    } else {
                                        String status = currentLeader != null
                                                ? String.format("Leading: %s (%d frames) • Remove hand to confirm", currentLeader, currentLeaderVotes)
                                                : "Low confidence (< 55%)";
                                        updateLiveMetrics(predicted, confidence, status);
                                    }
                                }
                            }
                        }
                    }
                }
            } catch (SocketTimeoutException ste) {
                System.err.println("ERROR: Python did not connect within timeout (30s).");
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

    static String commitSegment(Map<String, Integer> voteCounts, int minVotes) {
        String winner = null;
        int maxCount = -1;
        for (Map.Entry<String, Integer> entry : voteCounts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                winner = entry.getKey();
            }
        }
        if (winner != null && maxCount >= minVotes) {
            appendToSentence(winner);
            return winner;
        }
        return null;
    }
}
