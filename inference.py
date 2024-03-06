from model import detect_mask

def inference(input_path, output_path):

    with open(input_path, 'rb') as file:
        result = detect_mask(output_path, file)
        print(f"Result: {result}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on an image.")
    parser.add_argument("input", help="Path to the input image file")
    parser.add_argument("output", help="Path to save the output image with detections")

    args = parser.parse_args()

    inference(args.input, args.output)
