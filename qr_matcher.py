import cv2


def normalize_qr_data(data):
    """Normalize QR text before comparing it."""
    return data.strip().lower()


class QRMatcher:
    """Stores the target QR data and checks new QR codes against it."""

    def __init__(self):
        self.target_data = None

    def has_target(self):
        return self.target_data is not None

    def save_target(self, data):
        normalized_data = normalize_qr_data(data)

        if not normalized_data:
            return False

        self.target_data = normalized_data
        return True

    def reset_target(self):
        self.target_data = None

    def is_match(self, data):
        if self.target_data is None:
            return False

        return normalize_qr_data(data) == self.target_data


def get_qr_center(points):
    """Return the center point of a QR code box."""
    x_values = points[:, 0]
    y_values = points[:, 1]

    center_x = int(x_values.mean())
    center_y = int(y_values.mean())

    return center_x, center_y


def get_centering_error(qr_center, frame_shape):
    """Return error from the frame center to the QR center."""
    frame_height, frame_width = frame_shape[:2]
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2

    error_x = qr_center[0] - frame_center_x
    error_y = qr_center[1] - frame_center_y

    return error_x, error_y


def draw_center_point(frame, center, color=(0, 255, 0)):
    """Draw a dot at the center of a matching QR code."""
    cv2.circle(frame, center, 6, color, -1)


def draw_error_text(frame, error_x, error_y):
    """Display centering errors on the camera frame."""
    text = f"error_x: {error_x}  error_y: {error_y}"
    cv2.putText(
        frame,
        text,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
