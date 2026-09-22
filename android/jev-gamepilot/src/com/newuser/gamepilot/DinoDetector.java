package com.newuser.gamepilot;

import android.media.Image;
import java.nio.ByteBuffer;

/** Samples pixels without retaining or uploading a screenshot. */
final class DinoDetector {
    static final class Observation {
        final int width, height, groundY, nearHits, farHits;
        final boolean groundFound;
        Observation(int width, int height, int groundY, int nearHits, int farHits, boolean groundFound) {
            this.width = width; this.height = height; this.groundY = groundY;
            this.nearHits = nearHits; this.farHits = farHits; this.groundFound = groundFound;
        }
    }

    static Observation analyze(Image image) {
        Image.Plane p = image.getPlanes()[0];
        return analyzePixels(p.getBuffer(), image.getWidth(), image.getHeight(), p.getRowStride(), p.getPixelStride());
    }

    static Observation analyzePixels(ByteBuffer data, int w, int h, int row, int stride) {
        int base = brightness(data, row, stride, w * 83 / 100, h * 27 / 100);
        int bestY = -1, bestCount = 0;
        int dx = Math.max(3, w / 100), dy = Math.max(3, h / 170);
        for (int y = h * 34 / 100; y < h * 84 / 100; y += dy) {
            int count = 0;
            for (int x = w * 22 / 100; x < w * 84 / 100; x += dx) {
                if (Math.abs(brightness(data, row, stride, x, y) - base) >= 65) count++;
            }
            if (count > bestCount) { bestCount = count; bestY = y; }
        }
        int samples = (w * 62 / 100) / dx;
        boolean found = bestCount >= Math.max(12, samples * 47 / 100);
        if (!found) return new Observation(w, h, -1, 0, 0, false);
        int near = countObstacle(data, row, stride, w, h, bestY, base, 24, 44);
        int far = countObstacle(data, row, stride, w, h, bestY, base, 44, 65);
        return new Observation(w, h, bestY, near, far, true);
    }

    private static int countObstacle(ByteBuffer pixels, int row, int stride, int w, int h,
                                     int ground, int bg, int xStart, int xEnd) {
        int hits = 0;
        int from = Math.max(h * 25 / 100, ground - h * 10 / 100);
        int to = ground - Math.max(4, h / 150);
        for (int x = w * xStart / 100; x < w * xEnd / 100; x += Math.max(3, w / 120)) {
            for (int y = from; y < to; y += Math.max(3, h / 140)) {
                if (Math.abs(brightness(pixels, row, stride, x, y) - bg) >= 65) hits++;
            }
        }
        return hits;
    }

    private static int brightness(ByteBuffer pixels, int row, int stride, int x, int y) {
        int pos = y * row + x * stride;
        if (pos < 0 || pos + 2 >= pixels.limit()) return 255;
        return ((pixels.get(pos) & 255) * 3 + (pixels.get(pos + 1) & 255) * 6
                + (pixels.get(pos + 2) & 255)) / 10;
    }
}
