![logo](https://media.discordapp.net/attachments/855693880041144350/1126201907514462208/Pirl.png)

# Description

PIRL (Paint in Real Life) aims to provide a purely software-based alternative to interactive white boards, using homography, image recognition, and the mediapipe library to effectively emulate touch screen capabilities on any surface, be it a large TV or a projector's projected screen.

# Demo

https://github.com/rogitson/pirl/assets/79846026/6f188a0c-4994-4033-aac9-d9d232e74038

# Usage

```sh
git clone https://github.com/rogitson/pirl.git
pip install -r requirements.txt
git submodule update --init --recursive
```
# Known Issues

We did not have the opportunity to test all edge cases so there are several ways our application can crash.

Due to the inherent limitations of our idea and reliance on mediapipe's detection from the video feed, drawing using our app has some jittering.
