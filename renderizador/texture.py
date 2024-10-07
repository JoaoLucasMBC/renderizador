import numpy as np


class TextureHandler:

    mipmaps = []

    @staticmethod
    def get_texture(
        u, v, u_up, v_up, u_right, v_right
    ):
        #Config
        mipmaps = TextureHandler.mipmaps

        # Calculate the partial derivatives
        du_dx = u_right - u
        dv_dx = v_right - v

        du_dy = u_up - u
        dv_dy = v_up - v

        L = max([
            np.sqrt(du_dx**2 + dv_dx**2),
            np.sqrt(du_dy**2 + dv_dy**2)
        ])

        D = np.log2(L)

        # Floor and fractional part of D
        D_floor = int(np.floor(D))
        D_frac = D - D_floor  # Fractional part for interpolation

        # Ensure valid mipmap level
        D_floor = np.clip(D_floor, 0, len(mipmaps) - 1)
        D_ceil = np.clip(D_floor + 1, 0, len(mipmaps) - 1)

        # Sample both mipmap levels
        tex_x = u * mipmaps[D_floor].shape[1]
        tex_y = v * mipmaps[D_floor].shape[0]
        sample_D = TextureHandler._bilinearFilter(tex_x, tex_y, mipmaps[D_floor])

        tex_x_ceil = u * mipmaps[D_ceil].shape[1]
        tex_y_ceil = v * mipmaps[D_ceil].shape[0]
        sample_D_plus_1 = TextureHandler._bilinearFilter(tex_x_ceil, tex_y_ceil, mipmaps[D_ceil])

        # Perform linear interpolation between the two mipmap levels
        return (1 - D_frac) * sample_D + D_frac * sample_D_plus_1

    @staticmethod
    def calculate_uv(
        uv1, uv2, uv3,
        z1, z2, z3, z,
        alpha, beta, gamma
    ):
        u = (alpha * uv1[0]/z1 + beta * uv2[0]/z2 + gamma * uv3[0]/z3) * z
        v = 1 - (alpha * uv1[1]/z1 + beta * uv2[1]/z2 + gamma * uv3[1]/z3) * z

        return u,v

    @staticmethod
    def _bilinearFilter(tex_x, tex_y, texture):
        x0 = int(np.floor(tex_x))
        y0 = int(np.floor(tex_y))
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

        TextureHandler.mipmaps = mipmaps