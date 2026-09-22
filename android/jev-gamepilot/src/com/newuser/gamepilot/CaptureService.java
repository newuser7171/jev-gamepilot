package com.newuser.gamepilot;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.graphics.PixelFormat;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.SystemClock;
import android.util.DisplayMetrics;
import android.view.WindowManager;
import java.util.concurrent.atomic.AtomicBoolean;

public class CaptureService extends Service {
    private static final String CHANNEL = "gamepilot_capture";
    private MediaProjection projection;
    private VirtualDisplay display;
    private ImageReader reader;
    private HandlerThread worker;
    private Handler handler;
    private long lastJump;
    private long lastNotice;
    private long lastJevCall;
    private int jumps;
    private JevClient jev;
    private final AtomicBoolean jevBusy = new AtomicBoolean(false);
    private volatile String jevStatus = "Local reflex only";
    private volatile DinoDetector.Observation latest;
    static volatile String diagnostics = "Pilot stopped";

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null || "STOP".equals(intent.getAction()) || intent.getParcelableExtra("resultData") == null) {
            stopSelf(); return START_NOT_STICKY;
        }
        NotificationManager nm = getSystemService(NotificationManager.class);
        nm.createNotificationChannel(new NotificationChannel(CHANNEL, "Dino screen capture", NotificationManager.IMPORTANCE_LOW));
        String key = intent.getStringExtra("apiKey");
        jev = key == null || key.trim().isEmpty() ? null : new JevClient(key.trim());
        jevStatus = jev == null ? "Local reflex only · enter key for Jev" : "Jev ready · waiting for obstacle";
        diagnostics = "Capture started";
        if (projection != null) stopCapture();
        Notification notification = notification("Waiting for Chrome Dino · " + jevStatus);
        if (Build.VERSION.SDK_INT >= 29) startForeground(1, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
        else startForeground(1, notification);
        DisplayMetrics metrics = new DisplayMetrics();
        ((WindowManager)getSystemService(WINDOW_SERVICE)).getDefaultDisplay().getRealMetrics(metrics);
        int width = metrics.widthPixels, height = metrics.heightPixels;
        worker = new HandlerThread("DinoDetection"); worker.start(); handler = new Handler(worker.getLooper());
        reader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2);
        MediaProjectionManager mgr = (MediaProjectionManager)getSystemService(MEDIA_PROJECTION_SERVICE);
        Intent data = intent.getParcelableExtra("resultData");
        projection = mgr.getMediaProjection(intent.getIntExtra("resultCode", 0), data);
        projection.registerCallback(new MediaProjection.Callback() {
            @Override public void onStop() { stopSelf(); }
        }, handler);
        display = projection.createVirtualDisplay("Dino pilot", width, height, metrics.densityDpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR, reader.getSurface(), null, handler);
        reader.setOnImageAvailableListener(new ImageReader.OnImageAvailableListener() {
            @Override public void onImageAvailable(ImageReader source) { inspect(source); }
        }, handler);
        return START_NOT_STICKY;
    }

    private Notification notification(String message) {
        Intent stop = new Intent(this, CaptureService.class).setAction("STOP");
        PendingIntent stopIntent = PendingIntent.getService(this, 1, stop, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        return new Notification.Builder(this, CHANNEL)
                .setContentTitle("Dino pilot running")
                .setContentText(message)
                .setSmallIcon(android.R.drawable.ic_media_play)
                .addAction(android.R.drawable.ic_delete, "Stop", stopIntent).build();
    }

    private void inspect(ImageReader source) {
        Image image = source.acquireLatestImage();
        if (image == null) return;
        try {
            long now = SystemClock.elapsedRealtime();
            boolean chrome = TouchService.chromeVisible();
            if (TouchService.instance != null && chrome) {
                DinoDetector.Observation obs = DinoDetector.analyze(image);
                latest = obs;
                if (obs.groundFound && obs.nearHits >= 8) jumpIfReady(obs, now);
                if (jev != null && obs.groundFound && obs.farHits >= 5 && now - lastJevCall > 1300
                        && jevBusy.compareAndSet(false, true)) {
                    lastJevCall = now;
                    queryJev(obs);
                }
                diagnostics = obs.groundFound ? "Ground " + (obs.groundY * 100 / obs.height)
                        + "% · near " + obs.nearHits + " · ahead " + obs.farHits + " · jumps " + jumps
                        : "Chrome visible · ground not found";
            } else diagnostics = TouchService.instance == null ? "Enable Accessibility" : "Open Chrome Dino";
            if (now - lastNotice > 2100) {
                lastNotice = now;
                getSystemService(NotificationManager.class).notify(1, notification(diagnostics + " · " + jevStatus));
            }
        } catch (RuntimeException ignored) { } finally { image.close(); }
    }

    private void queryJev(final DinoDetector.Observation observation) {
        new Thread(new Runnable() { @Override public void run() {
            try {
                JevClient.Decision decision = jev.decide(observation);
                jevStatus = decision.status;
                // Network results are advisory; a stale answer never triggers a gesture.
                DinoDetector.Observation current = latest;
                if ("jump".equals(decision.action) && decision.confidence >= .65 && current != null
                        && current.groundFound && current.farHits >= 5 && TouchService.chromeVisible()) {
                    jumpIfReady(current, SystemClock.elapsedRealtime());
                }
            } finally { jevBusy.set(false); }
        }}, "JevDecision").start();
    }

    private synchronized void jumpIfReady(DinoDetector.Observation obs, long now) {
        if (now - lastJump < 550 || TouchService.instance == null || !TouchService.chromeVisible()) return;
        lastJump = now; jumps++;
        final TouchService touch = TouchService.instance;
        final int w = obs.width, h = obs.height;
        new Handler(getMainLooper()).post(new Runnable() { @Override public void run() { touch.jump(w, h); }});
    }
    private void stopCapture() {
        if (reader != null) { reader.setOnImageAvailableListener(null, null); reader.close(); reader = null; }
        if (display != null) { display.release(); display = null; }
        if (projection != null) { MediaProjection p = projection; projection = null; p.stop(); }
        if (worker != null) { worker.quitSafely(); worker = null; }
    }
    @Override public void onDestroy() { stopCapture(); jev = null; diagnostics = "Pilot stopped"; super.onDestroy(); }
    @Override public IBinder onBind(Intent intent) { return null; }
}
