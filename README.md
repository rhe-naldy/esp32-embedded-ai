This repository serves as an archive for a Master's degree project in Computer Science, titled "Development of Lightweight Models using Vegetation Indices to Identify Tomato Leaf Disease Deployed on a Microcontroller," published in Bina Nusantara University's repository and currently in progress for publication in a journal.
  
The header file [c_req_aug_vabc_VGGNet.h](https://github.com/rhe-naldy/esp32-embedded-ai/blob/main/c_req_aug_vabc_VGGNet.h) contains a model converted from a TensorFlow Lite (TFLITE) model, which was originally a lightweight VGG model coded from scratch by the author. After training the model from scratch, the model then undergo model pruning and model quantization in order to reduce the model size and fit it into the ESP32-S3-N16R8 microcontroller. Equipped with 8MB of PSRAM, the microcontroller was able to run the TFLite model, which required approximately only 2MB of PSRAM and an average runtime of 12.1 seconds.
  
Both the header file and the ESP32 firmware can be accessed within this repository under the filenames [c_req_aug_vabc_VGGNet.h](https://github.com/rhe-naldy/esp32-embedded-ai/blob/main/c_req_aug_vabc_VGGNet.h) and [ESP3S3_CAM.ino](https://github.com/rhe-naldy/esp32-embedded-ai/blob/main/ESP32S3_CAM.ino), respectively.

Below are detailed images regarding the project.
  
* System architecture diagram  
<img src="https://github.com/rhe-naldy/esp32-embedded-ai/blob/main/architecture_diagram.png?raw=true" width="256">
  
  
* Circuit diagram
<img src="https://github.com/rhe-naldy/esp32-embedded-ai/blob/main/circuit_diagram.png?raw=true" width="256">
  
  
* Image of the prototype
<img src="https://github.com/rhe-naldy/esp32-embedded-ai/blob/main/hardware_image.jpg?raw=true" width="256">
  
  
* Image of the inference
<img src="https://github.com/rhe-naldy/esp32-embedded-ai/blob/main/inference_image.jpg?raw=true" width="256">
