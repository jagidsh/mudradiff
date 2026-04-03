
def denormalize(image):
    return image * 0.5 + 0.5  # Scale [-1, 1] back to [0, 1]


