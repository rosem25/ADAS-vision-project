"""
Configuration settings for ADAS Lite Pipeline
"""

# Lane Detection
CANNY_LOW_THRESH = 50
CANNY_HIGH_THRESH = 150
HOUGH_THRESHOLD = 40
HOUGH_MIN_LINE_LENGTH = 40
HOUGH_MAX_LINE_GAP = 100
ROI_TOP_LEFT = (0.45, 0.6)      # (x_ratio, y_ratio)
ROI_TOP_RIGHT = (0.55, 0.6)
ROI_BOTTOM_LEFT = (0.1, 1.0)
ROI_BOTTOM_RIGHT = (0.9, 1.0)
LANE_SLOPE_THRESH = 0.5

# Object Detection (Fallback)
FALLBACK_AREA_THRESH = 500

# Warning Logic
WARNING_CENTER_TOLERANCE = 0.25 # +/- 25% from center
WARNING_AREA_THRESH = 0.03      # 3% of frame area

# Streaming
JPEG_QUALITY = 80
