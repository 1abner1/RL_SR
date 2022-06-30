from PIL import Image

WIDTH, HEIGHT = 50, 8
# ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")
ascii_char = list("************************ ")

def get_char_by_rgb(r, g, b, alpha=256):
    if alpha == 0:
        return ' '
    length = len(ascii_char)
    gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
    unit = 256.0 / length
    return ascii_char[int(gray / unit)]


def process_image(image_path, file_path='out.txt'):
    img = Image.open(image_path)
    img = img.resize((WIDTH, HEIGHT))
    width, height = img.size
    txt = ""
    for x in range(height):
        for y in range(width):
            txt += get_char_by_rgb(*img.getpixel((y, x)))
        txt += '\n'

    with open(file_path, 'w') as f:
        f.write(txt)
    print(txt)

def run():
    image_path, file_path = r'D:\RL_SR\algorithm\s2rlog\s2r1.png', r'D:\RL_SR\algorithm\s2rlog\1.txt'
    process_image(image_path, file_path)

if __name__ == '__main__':
    run()