package com.newuser.gamepilot;

import org.json.JSONObject;
import java.net.HttpURLConnection;
import java.net.URL;
import java.io.OutputStream;
import java.io.InputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

/** Direct TypeSafe System One HTTP client. The key exists in memory for this run only. */
final class JevClient {
    static final class Decision {
        final String action, status;
        final double confidence;
        Decision(String action, double confidence, String status) {
            this.action = action; this.confidence = confidence; this.status = status;
        }
    }
    private final String key;
    JevClient(String key) { this.key = key; }

    Decision decide(DinoDetector.Observation obs) {
        HttpURLConnection connection = null;
        try {
            JSONObject state = new JSONObject();
            state.put("game", "Chrome Dino");
            state.put("ground_detected", obs.groundFound);
            state.put("ground_y_fraction", obs.groundY / (double) obs.height);
            state.put("dark_pixels_near", obs.nearHits);
            state.put("dark_pixels_ahead", obs.farHits);
            state.put("description", "Near pixels are immediately in front of the runner; ahead pixels are farther right. A cluster of foreground pixels immediately above the ground can be an obstacle.");
            JSONObject criteria = new JSONObject();
            criteria.put("jump", "A ground obstacle is approaching the runner and a jump is appropriate now.");
            criteria.put("wait", "No clear approaching ground obstacle; avoid unnecessary taps.");
            JSONObject question = new JSONObject();
            question.put("type", "choice");
            question.put("instructions", "Select the immediate Chrome Dino action based only on the measured obstacle evidence. Do not assume an obstacle when evidence is absent.");
            question.put("criteria", criteria);
            JSONObject questions = new JSONObject().put("action", question);
            JSONObject body = new JSONObject().put("model", "jev-latest").put("state", state).put("questions", questions);

            connection = (HttpURLConnection) new URL("https://api.typesafe.ai/v1/systemone").openConnection();
            connection.setRequestMethod("POST");
            connection.setRequestProperty("Authorization", "Bearer " + key);
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setConnectTimeout(1700);
            connection.setReadTimeout(1700);
            connection.setDoOutput(true);
            byte[] bytes = body.toString().getBytes(StandardCharsets.UTF_8);
            try (OutputStream out = connection.getOutputStream()) { out.write(bytes); }
            int code = connection.getResponseCode();
            if (code != 200) return new Decision("wait", 0, "Jev HTTP " + code);
            String response;
            try (InputStream in = connection.getInputStream(); ByteArrayOutputStream out = new ByteArrayOutputStream()) {
                byte[] buffer = new byte[1024]; int n;
                while ((n = in.read(buffer)) >= 0) { out.write(buffer, 0, n); if (out.size() > 32768) throw new Exception("Response too large"); }
                response = out.toString("UTF-8");
            }
            JSONObject answer = new JSONObject(response).getJSONObject("answers").getJSONObject("action");
            String choice = answer.getString("choice");
            double confidence = answer.optDouble("confidence", 0);
            if (!"jump".equals(choice) && !"wait".equals(choice)) throw new Exception("Unexpected choice");
            return new Decision(choice, confidence, "Jev: " + choice + " (" + Math.round(confidence * 100) + "%)");
        } catch (Exception e) {
            return new Decision("wait", 0, "Jev unavailable: " + e.getClass().getSimpleName());
        } finally { if (connection != null) connection.disconnect(); }
    }
}
