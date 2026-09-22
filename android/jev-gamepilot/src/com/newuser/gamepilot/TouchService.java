package com.newuser.gamepilot;

import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.GestureDescription;
import android.graphics.Path;
import android.view.accessibility.AccessibilityEvent;

public class TouchService extends AccessibilityService {
    static volatile TouchService instance;
    static volatile String foregroundPackage = "";
    @Override protected void onServiceConnected() { super.onServiceConnected(); instance = this; }
    @Override public void onAccessibilityEvent(AccessibilityEvent event) {
        if (event.getPackageName() != null) foregroundPackage = event.getPackageName().toString();
    }
    @Override public void onInterrupt() { }
    @Override public void onDestroy() { if (instance == this) instance = null; foregroundPackage = ""; super.onDestroy(); }

    static boolean chromeVisible() { return foregroundPackage.equals("com.android.chrome")
            || foregroundPackage.equals("com.chrome.beta") || foregroundPackage.equals("com.chrome.dev"); }

    void jump(int width, int height) {
        Path path = new Path();
        path.moveTo(width * .62f, height * .45f);
        GestureDescription gesture = new GestureDescription.Builder()
                .addStroke(new GestureDescription.StrokeDescription(path, 0, 42)).build();
        dispatchGesture(gesture, null, null);
    }
}
