import requests

# image path
GAZE_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "./face.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
request = requests.post(GAZE_API_URL, files=payload).json()

# ensure the request was sucessful
if request["success"]:
	for (i, result) in enumerate(r["predictions"]):
		print(f"{request["pitch"]:.4f}, {request["yaw"]:.4f)}")

else:
	print("Request failed")