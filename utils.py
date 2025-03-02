import cv2

def resize_frame(frame, width=None, height=None):
    """Resize frame keeping aspect ratio"""
    if width and height:
        return cv2.resize(frame, (width, height))
    
    if width:
        h, w = frame.shape[:2]
        ratio = width / w
        return cv2.resize(frame, (width, int(h * ratio)))
    
    if height:
        h, w = frame.shape[:2]
        ratio = height / h
        return cv2.resize(frame, (int(w * ratio), height))
    
    return frame

def add_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                             font_scale=0.6, color=(255, 255, 255), thickness=1,
                             bg_color=(0, 0, 0), padding=5):
    """Add text with background to image"""
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate background rectangle dimensions
    x, y = position
    bg_x1 = x - padding
    bg_y1 = y - text_height - padding
    bg_x2 = x + text_width + padding
    bg_y2 = y + padding
    
    # Draw background rectangle
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    
    # Draw text
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    
    return frame