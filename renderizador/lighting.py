import numpy as np

class LightingHandler:

    @staticmethod
    def compute_lighting(light_sources, normal, material, point):
        diffuse_color = np.array(material['diffuseColor'])
        specular_color = np.array(material['specularColor'])
        shininess = material['shininess']

        normal = normal / np.linalg.norm(normal)

        v = np.array([0, 0, 1])

        final_light = np.zeros(3)

        for ls in light_sources:
            ambient_intensity = ls['ambientIntensity']
            color = np.array(ls['color'])
            intensity = ls['intensity']
            direction = np.array(ls['direction'])

            L = -direction

            ambient_light = ambient_intensity * diffuse_color # precisa de mais algo?
        
            dot = np.dot(normal, L)

            diffuse_light = intensity * diffuse_color * max(0, dot)
 
            h = L + v
            h = h / np.linalg.norm(h)

            specular_light = intensity * specular_color * max(0, np.dot(normal, h))**(shininess * 128)

            final_light += (ambient_light + diffuse_light + specular_light) * color
        
        final_light = np.clip(final_light, 0, 1)

        return final_light


    @staticmethod
    def compute_face_normal(p1, p2, p3):

        v1 = p2 - p1
        v2 = p3 - p1

        normal = np.cross(v1, v2)

        norm_length = np.linalg.norm(normal)
        if norm_length < 1e-6:
            return np.array([0, 0, 0]) 
        
        return normal / norm_length


    @staticmethod
    def compute_vertices_normal(vertices):

        vertices_normals = {tuple(vertices[i:i+3]): np.zeros(3) for i in range(0, len(vertices), 3)}

        for i in range(0, len(vertices), 9):
            x1, y1, z1, x2, y2, z2, x3, y3, z3 = vertices[i:i+9]
            p1, p2, p3 = np.array([x1, y1, z1]), np.array([x2, y2, z2]), np.array([x3, y3, z3])

            normal = LightingHandler.compute_face_normal(
                p1, p2, p3    
            )

            vertices_normals[tuple(p1)] += normal
            vertices_normals[tuple(p2)] += normal
            vertices_normals[tuple(p3)] += normal
        
        for vn in vertices_normals:
            vertices_normals[vn] = vertices_normals[vn] / np.linalg.norm(vertices_normals[vn])
        
        return vertices_normals