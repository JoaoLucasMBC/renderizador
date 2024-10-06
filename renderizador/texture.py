import numpy as np


class TextureHandler:

    @staticmethod
    def _bilinearFilter(tex_x, tex_y, texture):
        x0 = tex_x
        y0 = tex_y
        x1 = min(x0 + 1, texture.shape[1] - 1)
        y1 = min(y0 + 1, texture.shape[0] - 1)

        # Steps
        u_step = tex_x - x0
        v_step = tex_y - y0

        top_left = texture[x0, y0]
        top_right = texture[x1, y0]
        bottom_left = texture[x0, y1]
        bottom_right = texture[x1, y1]

        top_interp = (1 - u_step) * top_left + u_step * top_right
        bottom_interp = (1 - u_step) * bottom_left + u_step * bottom_right

        return (1 - v_step) * top_interp + v_step * bottom_interp
    
    @staticmethod
    def generate_mipmaps(texture):
        mipmaps = [texture]
        current_level = texture

        # Keep reducing size by half until reaching 1x1
        while current_level.shape[0] > 1 or current_level.shape[1] > 1:
            # Downsample the texture by averaging neighboring texels
            new_width = max(1, current_level.shape[1] // 2)
            new_height = max(1, current_level.shape[0] // 2)

            new_level = np.zeros((new_height, new_width, 3))

            for y in range(new_height):
                for x in range(new_width):
                    # Average the 4 texels from the previous mipmap level
                    new_level[y, x] = np.mean(current_level[2*y:2*y + 2, 2*x:2*x + 2], axis=(0, 1))

            mipmaps.append(new_level)
            current_level = new_level

        return mipmaps