// import libraries
#include <esp_heap_caps.h>
#include "esp_camera.h"

#include "soc/soc.h"           // disable brownout problems
#include "soc/rtc_cntl_reg.h"  // disable brownout problems

#define CAMERA_MODEL_ESP32S3_EYE
#include "camera_pins.h"

#include <TFT_eSPI.h>
#include "img_converters.h"

#include <Chirale_TensorFlowLite.h>

// #include "c_req_vabc_VGGNet.h"
#include "c_req_aug_vabc_VGGNet.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

constexpr int kTensorArenaSize = 3 * 1024 * 1024;
alignas(16) uint8_t *tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);

// resolution configuration
TFT_eSPI tft = TFT_eSPI();
int cam_col = 240;  // 400x296 camera image size
int cam_row = 240;
int tft_col = 240;  // the 240x240 is for TFT display
int tft_row = 240;
int mi_col = 256;  // mi stands for model_input
int mi_row = 256;
int image_byte = 3;
int camera_width[] = { 240, 400 };
int camera_height[] = { 240, 296 };

//  variables
static bool is_initialised = false;
uint8_t *snapshot_buf;  // points to the output of the captured image
sensor_t *s = NULL;
int colorCount = 0;
const float mean = 18.77009;
const float stdev = 22.366663;
unsigned long inferenceTime = 0;
unsigned long viTime = 0;
unsigned long minute = 0;
unsigned long second = 0;
unsigned long ms = 0;

// labels
const char *kCategoryLabels[10] = {
  "Healthy", "Early blight", "Late blight", "Bacterial spot", "Leaf mold", "Septoria leaf spot", "Target spot", "Mosaic virus", "Yellow leaf curl virus", "Spider mites"
};

bool camera_setup() {
  if (is_initialised) {
    return true;
  }
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_YUV422; //PIXFORMAT_YUV422; //PIXFORMAT_JPEG; //PIXFORMAT_RGB565;//
  config.frame_size = FRAMESIZE_240X240; //FRAMESIZE_240X240 or FRAMESIZE_CIF (this depends on whether we want to crop or add padding)
  config.jpeg_quality = 8; // lower number == high quality (range 10-63)
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

  esp_err_t err_cam = esp_camera_init(&config);
  if (err_cam != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err_cam);
    return false;
  } else {
    Serial.println("Camera init success");
  }

  s = esp_camera_sensor_get();
  // s->set_quality(s, 8);
  s->set_brightness(s, 0); // brightness set to normal
  s->set_contrast(s, 0); // contrast set to normal
  s->set_saturation(s, 0); // saturation set to normal
  s->set_special_effect(s, 0); // disabled special effects
  s->set_whitebal(s, 1); // disable (0) white balance, enable (1)
  s->set_awb_gain(s, 1); // disable (0) awb gain, enable (1)
  s->set_wb_mode(s, 2); // auto (0) white balance, 1 == sunny, 2 == cloudy, 3 == office, 4 == home
  s->set_exposure_ctrl(s, 1); // disable auto-exposure for balanced brightness, (1) enable auto-exposure
  s->set_aec2(s, 1); // disable advanced auto-exposure for finer control, (1) enable advanced auto-exposure
  s->set_ae_level(s, 1); // neutral auto-exposure level
  s->set_aec_value(s, 600); // adjust exposure value (lower for bright environments, higher for low-light) (0 - 1200)
  s->set_gain_ctrl(s, 0); // disable (0) auto-gain control, enable (1)
  s->set_agc_gain(s, 5); // 0 auto, 1-30 (lower = darker image) (higher == brightens dark image + introduce a bit of noise)
  s->set_gainceiling(s, (gainceiling_t)6); // higher gain ceiling allows more light sensitivity (0-6), lower it to reduce noise
  s->set_bpc(s, 1); // disable (0) bad pixel correction, enable (1)
  s->set_wpc(s, 1); // disable (0) white pixel correction, enable (1)
  s->set_raw_gma(s, 1); // disable (0) gamma correction , enable (1) for natural tones
  s->set_lenc(s, 1); // disable (0) lens correction to avoid distortions, enable (1)
  s->set_colorbar(s, 0); // disable color bar

  is_initialised = true;
  return true;
}

void rgb888_to_rgb565(uint8_t *rgb888, uint16_t *rgb565){
  int j = 0;
  for(int i = 0; i < cam_row * cam_col * 3; i+=3){
    uint8_t r = rgb888[i + 2];
    uint8_t g = rgb888[i + 1];
    uint8_t b = rgb888[i];

    rgb565[j++] = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
  }
}

bool camera_to_tft(uint8_t *rgb888) {  // draw_camera_to_tft
  uint16_t *rgb565 = (uint16_t *)heap_caps_malloc(cam_col * cam_row * sizeof(uint16_t), MALLOC_CAP_SPIRAM);

  rgb888_to_rgb565(rgb888, rgb565);

  tft.pushImage(0, 0, tft_col, tft_row, (uint16_t *)rgb565);
  free(rgb565);

  return true;
}

void print_output(String output, int conf) {  // draw_inference_to_tft
  tft.setTextColor(TFT_WHITE, TFT_BLUE);
  tft.setTextSize(2);
  tft.setCursor(0, 210);
  tft.print("class: " + output + " (" + String(conf) + "%)");
}

bool capture_image(uint8_t *output) {  //ei_camera_capture // this function displays image to tft and then convert image //uint32_t w, uint32_t h,
  if (!is_initialised) {
    Serial.println("ERROR: Camera is not initialised (capture_image)...");
    return false;
  }

  camera_fb_t *fb = esp_camera_fb_get();

  if (!fb) {
    Serial.println("ERROR: Camera capture failed (capture_image)...");
    return false;
  }

  bool converted_img = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_YUV422, snapshot_buf);

  if (!converted_img) {
    Serial.println("ERROR: Image conversion failed (capture_image)...");
    return false;
  }
  
  if (!camera_to_tft(snapshot_buf)) {
    Serial.println("ERROR: Failed to display image on TFT (capture_image)...");
    return false;
  }

  esp_camera_fb_return(fb);

  return true;
}

void calc_vi(uint8_t *image) {
  int8_t pad_val = (int8_t) roundf(((((0.590 + 0.193) * (0 - (0.095 * 0) - (0.821 * 255))) - mean) / stdev) / input->params.scale + input->params.zero_point);  // blue padding

  for (int y = 0; y < mi_row; y++) {
    for (int x = 0; x < mi_col; x++) {
      if (y < 8 || y >= 248 || x < 8 || x >= 248) {
        input->data.int8[y * mi_row + x] = pad_val;
      } else {
        x += 239;
      }
    }
  }

  Serial.println("(calc_vi) Successfully padded the image");

  uint8_t r;
  uint8_t g;
  uint8_t b;
  float temp;
  int8_t quantized_input;

  Serial.println("(calc_vi) Variables declared...");

  // Serial.print("DATA: ");
  // for (size_t i = 0; i < cam_row * cam_col * 3; i++){
  //   Serial.print(image[i]); Serial.print(",");
  // }
  // Serial.println();

  // Serial.print("DATA: ");
  for (size_t y = 0; y < cam_row; y++) {
    for (size_t x = 0; x < cam_col; x++) {
      size_t idx = (y * cam_row + x) * 3; // correct format is BGR (i've checked)
      r = image[idx + 2];
      g = image[idx + 1];
      b = image[idx];

      float vodgi = (0.590 + 0.193) * ((float)g - (0.095 * (float)r) - (0.821 * (float)b));
      temp = ((vodgi - mean) / stdev);
      // Serial.print(temp, 6); Serial.print(",");
      quantized_input = (int8_t) roundf((temp / input->params.scale) + input->params.zero_point);
      input->data.int8[(y + 8) * mi_row + (x + 8)] = quantized_input;
    }
  }
  // Serial.println();

  Serial.println("(calc_vi) index calculated...");
}

int calc_conf(int8_t curr_conf){
  float f_val = (float) ((curr_conf - (-128)) / 255);
  Serial.println(f_val);
  float conf = round((f_val) * 100);

  return (int) conf;
}

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); //disable brownout detector
  Serial.begin(115200);
  Serial.setDebugOutput(true);

  if (!psramFound()) {
    Serial.println("PSRAM not found!");
  } else {
    Serial.println("PSRAM is available.");
  }

  if (!tensor_arena) {
    Serial.println("Failed to allocate tensor arena in PSRAM!");
  } else {
    Serial.printf("Tensor arena allocated: %d bytes in PSRAM\n", kTensorArenaSize);
  }


  Serial.println("Initializing Camera...");
  if (!camera_setup()) {
    Serial.println("Failed to initialize camera...");
  } else {
    Serial.println("Camera initialized...");
  }

  delay(1000);
  Serial.println("Initializing TFT Screen...");
  tft.init();
  tft.setSwapBytes(true);
  Serial.println("TFT Screen initialized...");
  delay(1000);
  tft.fillScreen(TFT_BLUE);

  snapshot_buf = (uint8_t *)heap_caps_malloc(cam_col * cam_row * image_byte, MALLOC_CAP_SPIRAM);  // snapshot buffer

  if (snapshot_buf == nullptr) {  // check the state of the snapshot buffer
    Serial.println("ERROR: Failed to allocate snapshot buffer...");
    delay(100);
  }

  Serial.println("Snapshot buffer successfully allocated...");

  model = tflite::GetModel(vggnet_model);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model provided and schema version are not equal!");
    while (true); // stop program here
  }

  Serial.println("Model successfully initialized...");

  static tflite::AllOpsResolver resolver;
  Serial.println("Resolver established...");
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  Serial.println("Establishing interpreter...");
  interpreter = &static_interpreter;
  Serial.println("Interpreter established...");
  // allocate memory for model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while(true); // stop program here
  }

  Serial.println("Tensors successfully allocated...");

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Serial.print("input scale: "); Serial.println(input->params.scale);
  // Serial.print("zero point: "); Serial.println(input->params.zero_point);

  Serial.println("Model input and output tensors obtained...");

  delay(1000);

  Serial.printf("Free Heap: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());

  delay(1000);

  Serial.println("Entering loop function...");
}

void loop() {
  // might remove the above code to blue screen that says steady your camera
  unsigned long current_time = millis();
  Serial.println("Taking a picture...");
  while (millis() - current_time < 15000) {  // display image for 15 seconds
    if (!capture_image(snapshot_buf)) {     // display tft and convert image (status 0)
      Serial.println("ERROR: Failed to convert image...");
      free(snapshot_buf);
      return;
    }
    tft.setTextColor(TFT_WHITE, TFT_BLUE);
    tft.setTextSize(2);
    tft.setCursor(0, 0);
    tft.printf("%d", int(15 - (millis() - current_time) / 1000));
    delay(100);
  }
  Serial.println("Picture successfully taken...");

  // this point here will be filled with classification stuff
  // calculate vi
  viTime = millis();
  calc_vi(snapshot_buf);
  viTime = millis() - viTime;
  minute = (int)round(viTime / 60000);
  second = (viTime/1000) % 60;
  ms = viTime % 1000;
  Serial.printf("Index calculated successfully with a time of: %d minutes %d.%d seconds\n", minute, second, ms);
  Serial.println("Successfully assigned quantized data into the input tensor...");
  Serial.println("Invoking model...");

  inferenceTime = millis();
  // Run inference, and report if an error occurs
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }
  Serial.println("Invoke success...");

  int idx = 0;
  int8_t max_conf = output->data.int8[idx];
  int8_t curr_conf;

  for (int i = 0; i < 10; i++){
    Serial.print(kCategoryLabels[i]);
    Serial.print(": ");
    Serial.print(output->data.int8[i]);
    Serial.println();
  }

  for (int i = 0; i < 10; i++) {
    curr_conf = output->data.int8[i];

    if (max_conf < curr_conf) {  // if current conf > max conf1 then maxconf1 = curr conf. then, demote maxconf1 and 2
      idx = i;
      max_conf = curr_conf;
    }
  }

  Serial.printf("Disease detected as %s with a confidence of (%d%)\n", kCategoryLabels[idx], calc_conf(max_conf));
  print_output(kCategoryLabels[idx], calc_conf(max_conf));

  inferenceTime = millis() - inferenceTime;
  minute = (int)round(inferenceTime / 60000);
  second = (inferenceTime/1000) % 60;
  ms = inferenceTime % 1000;

  Serial.printf("Inference time: %d minutes %d.%d seconds\n", minute, second, ms);

  delay(1000);  //delay 1s

  Serial.printf("Free Heap: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());

  delay(10000);  //delay 10s
}
