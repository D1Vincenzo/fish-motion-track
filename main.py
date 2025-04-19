import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

clicked_points = []
point_labels = ["Calibration 1", "Calibration 2", "Fixed Point", "Track Point"]

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 3:
            clicked_points.append((x, y))
            print(f"Added point: {point_labels[len(clicked_points) - 1]} → ({x}, {y})")
        else:
            clicked_points.append((x, y))
            print(f"Added track point {len(clicked_points) - 3} → ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN and len(clicked_points) > 0:
        removed = clicked_points.pop()
        print(f"Removed last point: {removed}")

def draw_selection_ui(image, points, reverse=False):
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, pt in enumerate(points):
        if i < 3:
            label = point_labels[i]
            color = (255, 0, 255) if i < 2 else (0, 0, 255)
        else:
            label = f"Track {i - 2}"
            color = (0, 255, 0)

        cv2.circle(img, pt, 6, color, -1)
        text_size = cv2.getTextSize(label, font, 0.9, 2)[0]
        text_x = pt[0] + 15 if reverse else pt[0] - text_size[0] - 10
        text_y = pt[1] - 10
        cv2.putText(img, label, (text_x, text_y), font, 0.9, color, 2)

    if len(points) >= 2:
        cv2.line(img, points[0], points[1], (200, 200, 0), 2)
        calib_label = "Calibration Distance = 85mm"
        text_size = cv2.getTextSize(calib_label, font, 0.9, 2)[0]
        text_x = points[0][0] + 15 if reverse else points[0][0] - text_size[0] - 10
        text_y = points[0][1] - 35
        cv2.putText(img, calib_label, (text_x, text_y), font, 0.9, (200, 200, 0), 2)

    if len(points) < 3:
        tip_text = f"Click Point {len(points)+1}/3: {point_labels[len(points)]} (Right-click to undo)"
    else:
        tip_text = f"Click Track Point {len(points)-2} (Right-click to undo)"

    cv2.putText(img, tip_text, (20, 40), font, 0.9, (0, 255, 255), 2)

    return img


def enhance_contrast(gray, gamma=1.2, clip_limit=2.0):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(gray, table)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gamma_corrected)

def compute_signed_angle(vec, ref_vec):
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    ref_norm = ref_vec / (np.linalg.norm(ref_vec) + 1e-8)
    dot = np.dot(ref_norm, vec_norm)
    cross = ref_norm[0] * vec_norm[1] - ref_norm[1] * vec_norm[0]
    angle = np.arctan2(cross, dot)
    return np.degrees(angle)

def track_single_point(video_path, selection_frame_index_sec=3):
    global clicked_points
    selection_frame_index = int(selection_frame_index_sec * 120)  # adjust frame rate as needed

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if selection_frame_index >= frame_count:
        print(f"Error: Frame index {selection_frame_index} exceeds video length {frame_count}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, selection_frame_index)
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to grab frame for point selection.")
        return

    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    output_csv = f"{video_basename}_tracking_data.csv"

    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", click_event, first_frame)

    print("Please click the following 4 points in order:")
    print("1. Calibration Point 1 (top of reference)")
    print("2. Calibration Point 2 (85 mm vertically from Point 1)")
    print("3. Fixed Point (base of the stick)")
    print("4. Track Point (tip of the stick)")
    print("You can RIGHT-CLICK to undo the last point.")

    while True:
        reverse_preview = False
        if len(clicked_points) >= 4:
            reverse_preview = clicked_points[3][0] > clicked_points[2][0]

        ui_frame = draw_selection_ui(first_frame, clicked_points, reverse=reverse_preview)

        cv2.imshow("Select Points", ui_frame)
        if cv2.waitKey(30) & 0xFF == 13 and len(clicked_points) >= 4:
            break
    cv2.destroyAllWindows()

    reverse_horizontal = clicked_points[3][0] > clicked_points[2][0]
    print(f"↔ Reverse direction? {'Yes' if reverse_horizontal else 'No'}")

    # Save annotated image
    annotated_image = draw_selection_ui(first_frame, clicked_points, reverse=reverse_horizontal)
    save_img = f"{video_basename}_selected_points.png"
    cv2.imwrite(save_img, annotated_image)
    print(f"✅ Saved point selection image: {save_img}")

    # Set video to next frame for tracking
    cap.set(cv2.CAP_PROP_POS_FRAMES, selection_frame_index + 1)

    calibration_pt1, calibration_pt2 = clicked_points[0], clicked_points[1]
    fixed_point = clicked_points[2]
    track_points = np.array([[pt] for pt in clicked_points[3:]], dtype=np.float32)

    pixel_distance = abs(calibration_pt1[1] - calibration_pt2[1])
    mm_per_pixel = 85.0 / pixel_distance
    print(f"Pixel height diff: {pixel_distance} px → {mm_per_pixel:.4f} mm/pixel")

    old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    old_gray = enhance_contrast(old_gray)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    y_trajectories = [[] for _ in range(len(track_points))]
    angle_trajectories = [[] for _ in range(len(track_points))]
    time_stamps = []
    frame_idx = 0

    ref_vec = np.array([1, 0]) if reverse_horizontal else np.array([-1, 0])

    with open(output_csv, mode='w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        header = ['time (s)']
        for i in range(len(track_points)):
            header += [f'y_offset_{i} (mm)', f'angle_{i} (deg)']
        writer.writerow(header)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_gray = enhance_contrast(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, track_points, None, **lk_params)
            good_new = p1[st == 1].reshape(-1, 2)

            if len(good_new) != len(track_points):
                print("Tracking failed!")
                break

            frame_data = [frame_idx / fps]
            for i, pt in enumerate(good_new):
                vec = pt - np.array(fixed_point)
                y_offset_px = fixed_point[1] - pt[1]
                y_offset_mm = y_offset_px * mm_per_pixel
                angle_deg = compute_signed_angle(vec, ref_vec)

                y_trajectories[i].append(y_offset_mm)
                angle_trajectories[i].append(angle_deg)
                frame_data += [f"{y_offset_mm:.3f}", f"{angle_deg:.3f}"]

                cv2.circle(frame, tuple(pt.astype(int)), 5, (0, 255, 0), -1)
                cv2.line(frame, tuple(map(int, fixed_point)), tuple(pt.astype(int)), (0, 255, 255), 2)

            writer.writerow(frame_data)
            time_stamps.append(frame_idx / fps)

            for i in range(len(good_new)):
                y_val = y_trajectories[i][-1]
                angle_val = angle_trajectories[i][-1]
                cv2.putText(frame, f"T{i+1}: Y={y_val:.1f}mm A={angle_val:.1f}deg",
                            (10, 40 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)

            cv2.imshow("Tracking Multiple Points", frame)
            if cv2.waitKey(30) & 0xFF == 27:
                break

            old_gray = frame_gray.copy()
            track_points = good_new.reshape(-1, 1, 2)
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Tracking CSV saved to: {output_csv}")

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for i in range(len(track_points)):
        ax[0].plot(time_stamps, y_trajectories[i], label=f'Track {i+1} Y')
        ax[1].plot(time_stamps, angle_trajectories[i], label=f'Track {i+1} Angle')

    ax[0].set_ylabel("Y Offset (mm)")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].set_ylabel("Angle (deg)")
    ax[1].set_xlabel("Time (s)")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Prompt user input
    video_id = input("Enter video ID (e.g., 4130): ").strip()
    try:
        time_sec = float(input("Enter time (in seconds) for point selection: ").strip())
    except ValueError:
        print("Invalid time input. Please enter a number.")
        exit()

    video_path = f"videos/DSC_{video_id}.MOV"
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        exit()

    track_single_point(video_path, selection_frame_index_sec=time_sec)
