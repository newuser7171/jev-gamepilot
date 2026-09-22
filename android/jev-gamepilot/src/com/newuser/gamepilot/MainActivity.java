package com.newuser.gamepilot;

import android.app.Activity;
import android.content.Intent;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Bundle;
import android.Manifest;
import android.content.pm.PackageManager;
import android.provider.Settings;
import android.view.Gravity;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.EditText;
import android.view.View;
import android.text.InputType;

public class MainActivity extends Activity {
    private static final int CAPTURE_REQUEST = 7;
    private TextView status;
    private EditText apiKey;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout layout = new LinearLayout(this);
        layout.setPadding(32, 48, 32, 32);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setGravity(Gravity.CENTER_HORIZONTAL);
        TextView title = new TextView(this);
        title.setText("GamePilot · Jev Dino");
        title.setTextSize(24);
        layout.addView(title);
        status = new TextView(this);
        status.setPadding(0, 24, 0, 24);
        status.setText("Chrome Dino preview. Enter your TypeSafe key to enable Jev decisions. The app also uses a fast on-device jump reflex.");
        layout.addView(status);
        apiKey = new EditText(this);
        apiKey.setHint("TypeSafe API key (optional)");
        apiKey.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);
        apiKey.setSingleLine(true);
        layout.addView(apiKey);
        TextView privacy = new TextView(this);
        privacy.setText("The key is used for this session only. Jev receives obstacle measurements, not screenshots.");
        layout.addView(privacy);
        Button testKey = new Button(this);
        testKey.setText("Test Jev key");
        testKey.setOnClickListener(new View.OnClickListener() { @Override public void onClick(View v) {
            final String key = apiKey.getText().toString().trim();
            if (key.isEmpty()) { status.setText("Enter a TypeSafe API key to test Jev."); return; }
            status.setText("Contacting Jev...");
            new Thread(new Runnable() { @Override public void run() {
                final JevClient.Decision answer = new JevClient(key).decide(
                        new DinoDetector.Observation(100, 100, 0, 0, 0, false));
                runOnUiThread(new Runnable() { @Override public void run() {
                    status.setText(answer.status + (answer.status.startsWith("Jev:")
                            ? " · connection works" : " · check the key or network"));
                }});
            }}, "JevConnectionTest").start();
        }});
        layout.addView(testKey);
        Button access = new Button(this);
        access.setText("1. Enable touch control");
        access.setOnClickListener(new View.OnClickListener() { @Override public void onClick(View v) {
            startActivity(new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS));
        }});
        layout.addView(access);
        Button start = new Button(this);
        start.setText("2. Start pilot with Jev");
        start.setOnClickListener(new View.OnClickListener() { @Override public void onClick(View v) {
            if (TouchService.instance == null) {
                status.setText("Enable GamePilot Dino Preview under Accessibility > Installed apps, then return here.");
                return;
            }
            MediaProjectionManager manager = (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
            startActivityForResult(manager.createScreenCaptureIntent(), CAPTURE_REQUEST);
        }});
        layout.addView(start);
        Button stop = new Button(this);
        stop.setText("Stop pilot");
        stop.setOnClickListener(new View.OnClickListener() { @Override public void onClick(View v) {
            stopService(new Intent(MainActivity.this, CaptureService.class)); status.setText("Pilot stopped.");
        }});
        layout.addView(stop);
        TextView help = new TextView(this);
        help.setPadding(0, 24, 0, 0);
        help.setText("Choose Entire screen when Android asks what to capture. Open chrome://dino in Chrome and start the game. The notification shows ground detection, jump count, and Jev status. Stop there or with this button.");
        layout.addView(help);
        setContentView(layout);
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 22);
        }
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != CAPTURE_REQUEST || resultCode != RESULT_OK || data == null) return;
        Intent service = new Intent(this, CaptureService.class);
        service.putExtra("apiKey", apiKey.getText().toString().trim());
        service.putExtra("resultCode", resultCode);
        service.putExtra("resultData", data);
        if (Build.VERSION.SDK_INT >= 26) startForegroundService(service); else startService(service);
        status.setText("Pilot running. Open chrome://dino in Chrome. Detection status appears in the notification.");
    }
}
