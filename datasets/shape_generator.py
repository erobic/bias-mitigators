import os
import numpy as np
import cv2


class ShapeGenerator():
    def __init__(self, dim=32, padding=8):
        self.dim = dim
        self.padded_dim = dim - padding * 2
        self.padding = padding
        self.shapes = ['circle', 'right_triangle', 'obtuse_triangle', 'square', 'parallelogram', 'kite',
                       'pentagon', 'semi_circle', 'plus', 'arrow']

    def init_image(self):
        return np.zeros((self.padded_dim, self.padded_dim, 3), np.uint8)

    def generate_polygon(self, vertices, image=None, color=(255, 255, 255)):
        if image is None:
            image = self.init_image()

        pts = vertices.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=1)
        # fill it
        cv2.fillPoly(image, [pts], color=color)
        return image

    def generate_circle(self):
        center = (int(self.padded_dim / 2), int(self.padded_dim / 2))
        radius = int(self.padded_dim / 2)
        image = self.init_image()
        cv2.circle(image, center, radius, (255, 255, 255), thickness=-1)
        return image

    def generate_semi_circle(self):
        circle = self.generate_circle()
        circle[:, int(self.padded_dim / 2):] = 0
        return circle

    def generate_right_triangle(self):
        vertices = np.array(
            [[0, 0],
             [0, self.padded_dim],
             [self.padded_dim, self.padded_dim]], np.int32)
        return self.generate_polygon(vertices)

    def generate_obtuse_triangle(self):
        vertices = np.array([[0, 0],
                             [self.padded_dim / 3, self.padded_dim],
                             [self.padded_dim, self.padded_dim]], np.int32)
        return self.generate_polygon(vertices)

    def generate_square(self):
        vertices = np.array([[0, 0],
                             [0, self.padded_dim],
                             [self.padded_dim, self.padded_dim],
                             [self.padded_dim, 0]], np.int32)
        return self.generate_polygon(vertices)

    def generate_parallelogram(self):
        vertices = np.array([[self.padded_dim / 3, 0],
                             [0, self.padded_dim],
                             [self.padded_dim * 2 / 3, self.padded_dim],
                             [self.padded_dim, 0]], np.int32)
        return self.generate_polygon(vertices)

    def generate_kite(self):
        vertices = np.array([[self.padded_dim / 2, 0],
                             [0, self.padded_dim / 3],
                             [self.padded_dim / 2, self.padded_dim],
                             [self.padded_dim, self.padded_dim / 3]], np.int32)
        return self.generate_polygon(vertices)

    def generate_pentagon(self):
        vertices = np.array([[self.padded_dim / 2, 0],
                             [0, self.padded_dim / 3],
                             [self.padded_dim / 3, self.padded_dim],
                             [self.padded_dim * 2 / 3, self.padded_dim],
                             [self.padded_dim, self.padded_dim / 3]], np.int32)
        return self.generate_polygon(vertices)

    def generate_hexagon(self):
        vertices = np.array([[self.padded_dim / 4, 0],
                             [0, self.padded_dim / 2],
                             [self.padded_dim / 4, self.padded_dim],
                             [self.padded_dim * 3 / 4, self.padded_dim],
                             [self.padded_dim, self.padded_dim / 2],
                             [self.padded_dim * 3 / 4, 0]], np.int32)
        return self.generate_polygon(vertices)

    def generate_plus(self):
        image = self.init_image()
        vert_rect = np.array([[self.padded_dim / 3, 0],
                              [self.padded_dim / 3, self.padded_dim],
                              [self.padded_dim * 2 / 3, self.padded_dim],
                              [self.padded_dim * 2 / 3, 0]], np.int32)
        hor_rect = np.array([[0, self.padded_dim / 3],
                             [0, self.padded_dim * 2 / 3],
                             [self.padded_dim, self.padded_dim * 2 / 3],
                             [self.padded_dim, self.padded_dim / 3]], np.int32)
        image = self.generate_polygon(vert_rect, image=image)
        return self.generate_polygon(hor_rect, image=image)

    def generate_arrow(self):
        vertices = np.array([[0, self.padded_dim / 4],
                             [0, self.padded_dim * 3 / 4],
                             [self.padded_dim * 2 / 3, self.padded_dim * 3 / 4],
                             [self.padded_dim, self.padded_dim / 2],
                             [self.padded_dim * 2 / 3, self.padded_dim / 4]], np.int32)
        return self.generate_polygon(vertices)

    def generate(self, shape):
        method = getattr(self, 'generate_' + shape)
        shape = method()
        img = np.zeros((self.dim, self.dim, 3))
        img[self.padding:self.dim - self.padding, self.padding:self.dim - self.padding] = shape
        return img


if __name__ == "__main__":
    generator = ShapeGenerator()
    save_dir = '/tmp/shapes'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for s in generator.shapes:
        img = generator.generate(s)
        cv2.imwrite(save_dir + f'/{s}.jpg', img)
        print(f"Saved {s}")
