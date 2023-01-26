import cv2
import mediapipe as mp
import wandb
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

wandb.init(project = "Bounding Boxes detection")

# For static images:

with mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            model_name='Shoe') as objectron:
  for i in range(4):
    image = cv2.imread(f"input/k{i}.color.jpg")
    # Convert the BGR image to RGB and process it with MediaPipe Objectron.
    results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw box landmarks.
    if not results.detected_objects:
      print(f'No box landmarks detected on input/k{i}.color.jpg')
      continue
    print(f'Box landmarks of input/k{i}.color.jpg:')
    annotated_image = image.copy()
    for detected_object in results.detected_objects:
      mp_drawing.draw_landmarks(
          annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
      mp_drawing.draw_axis(annotated_image, detected_object.rotation,
                           detected_object.translation)
      cv2.imwrite('/tmp/annotated_image' + str(i) + '.png', annotated_image)

    images = wandb.Image(annotated_image, caption="Image with predicted 3D bounding boxes")
    wandb.log({"Image" : images})