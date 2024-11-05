import segmentation
import feature_extraction
import classification
import preprocessing

import cv2
import argparse
import sys
import time


def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process an image or video file.")

    # Add arguments for specifying either video or image path
    parser.add_argument("--video", type=str, help="Path to the video file to process.")
    parser.add_argument("--image", type=str, help="Path to the image file to process.")

    # Parse the arguments
    args = parser.parse_args()
    classification.initClassifiers()

    # Check which argument is provided and call the appropriate function
    if args.video and not args.image:
        video_path = args.video
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        while cap.isOpened():
            # Read a frame from the video
            ret, frame = cap.read()

            if not ret:
                print("Reached end of video or encountered a read error.")
                break

            # Determine label using classifier
            segmented_frame = segmentation.process_image(frame)

            if not preprocessing.contains_red_circle(frame):
                cv2.putText(frame, "Unknown", (20, 20), 0, 1, 0)

            else:
                x = feature_extraction.get_features(segmented_frame)

                result, probability = classification.useClassifiers(x)

                cv2.putText(
                    frame,
                    str(result) + ", probability: " + str(probability),
                    (20, 20),
                    0,
                    1,
                    0,
                )

            # Display the processed frame (for debugging, if needed)
            cv2.imshow("sign-classification", frame)

            # Exit the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release the video capture object and close display window
        cap.release()
        cv2.destroyAllWindows()
    elif args.image and not args.video:
        start_time = time.time_ns()
        frame = cv2.imread(args.image)
        segmented_frame = segmentation.process_image(frame)

        features = feature_extraction.get_features(segmented_frame)

        if not preprocessing.contains_red_circle(frame):
            print("Unknown")

        else:
            result = classification.useClassifiers(features)
            print(
                "Image processing time ms: " + str((time.time_ns() - start_time) / 1e6)
            )

            for entry in result:
                print(entry)
    else:
        print("Error: Please specify either --video or --image, but not both.")
        sys.exit(1)


if __name__ == "__main__":
    main()
