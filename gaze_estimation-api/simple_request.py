import requests
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='request test for gaze estimation api')
    parser.add_argument(
        '--img', dest='img_path', help='image path',
        default="./IU.jpg", type=str)

    args = parser.parse_args()
    return args

# image path
if __name__ == '__main__':
    args = parse_args()
    GAZE_API_URL = "http://127.0.0.1:5000/predict"
    IMAGE_PATH = args.img_path


    # load the input image and construct the payload for the request
    image = open(IMAGE_PATH, "rb").read()
    payload = {"image": image}

    # submit the request
    request = requests.post(GAZE_API_URL, files=payload).json()

    # ensure the request was sucessful
    if request["success"]:
        for (i, result) in enumerate(request["predictions"]):
            print(f"{result['pitch']:.4f}, {result['yaw']:.4f}")

    else:
        print("Request failed")