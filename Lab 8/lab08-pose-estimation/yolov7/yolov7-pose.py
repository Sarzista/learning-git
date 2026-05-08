import time
import torch
import cv2
import numpy as np
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts


def pose_video(frame):
    img = letterbox(frame, input_size, stride=64, auto=True)[0]

    img = transforms.ToTensor()(img)
    img = torch.tensor(np.array([img.numpy()]))
    img = img.to(device)

    with torch.no_grad():
        t1 = time.time()
        output, _ = model(img)
        t2 = time.time()

        fps_value = 1 / (t2 - t1)

        output = non_max_suppression_kpt(
            output,
            0.25,
            0.65,
            nc=1,
            nkpt=17,
            kpt_label=True
        )

        output = output_to_keypoint(output)

    nimg = img[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    return nimg, fps_value


input_size = 256

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print("Selected Device:", device)

weights = torch.load(
    "yolov7-w6-pose.pt",
    map_location=device,
    weights_only=False
)

model = weights["model"]
model.float().eval()
model.to(device)

video_name = "skydiving"
video_path = f"../media/{video_name}.mp4"
save_name = video_name

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {video_path}")

out = None

print("Running YOLOv7 Pose on:", video_path)
print("Press q on the video window to stop.")

if __name__ == "__main__":
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Unable to read frame. Exiting.")
            break

        img, fps_value = pose_video(frame)

        cv2.putText(
            img,
            "FPS : {:.2f}".format(fps_value),
            (200, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            "YOLOv7",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        if out is None:
            out_h, out_w, _ = img.shape
            out = cv2.VideoWriter(
                f"{save_name}_yolov7_pose.avi",
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                10,
                (out_w, out_h)
            )

        cv2.imshow("Output", img)
        out.write(img)

        key = cv2.waitKey(1)

        if key == ord("q"):
            print("Stopped by user.")
            break

    cap.release()

    if out is not None:
        out.release()

    cv2.destroyAllWindows()

    print(f"Output saved as: {save_name}_yolov7_pose.avi")