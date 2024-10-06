#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante
    sampling = 4

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far
        
        GL.viewpoint_matrix = np.identity(4)
        GL.perspective_matrix = np.identity(4)
        
        GL.transformation_stack = [np.identity(4)]

        GL.sample_frame_buffer = np.zeros((GL.sampling * GL.height, GL.sampling * GL.width, 3), dtype=np.uint8)
        
        GL.z_buffer = np.full((GL.height * GL.sampling, GL.width * GL.sampling), np.inf)

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).

        # Para cada um dos pontos, o desenha na tela
        for i in range(0, len(point), 2):
            pos_x = int(point[i])
            pos_y = int(point[i + 1])

            # Checando se não está tentando desenhar fora da tela
            if pos_x >= 0 and pos_x < GL.width and pos_y >= 0 and pos_y < GL.height:
                color = [int(255 * colors['emissiveColor'][i]) for i in range(len(colors['emissiveColor']))]
                gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, color)

        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polyline2D
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        color = [int(255 * colors['emissiveColor'][i]) for i in range(len(colors['emissiveColor']))]
        
        # Pega sempre o ponto atual e proximo
        for i in range(0, len(lineSegments) - 2, 2):
            x1, y1 = lineSegments[i], lineSegments[i + 1]
            x2, y2 = lineSegments[i + 2], lineSegments[i + 3]
            
            # Usa as diferencas entre os x e os y para saber quem está crescendo mais,
            # o que determina quem "manda" no algoritmo
            dx = abs(x2 - x1) 
            dy = abs(y2 - y1)

            # Determina se é crescente ou decresente em cada coordenada
            slope_x = 1 if x2 > x1 else -1
            slope_y = 1 if y2 > y1 else -1
            
            # Caso onde o x cresce mais
            if dx > dy:
                # Calcula o coef ang da reta. Caso seja uma reta vertical, o coef eh 1
                # para pintar todos os pixels entre os dois valores de y 
                slope = dy/dx if dx != 0 else 1
                err = 0

                # Enquanto nao cheguei no ultimo ponto
                while int(x1) != int(x2):
                    # Pinta apenas se estiver dentro da tela
                    if x1 >= 0 and x1 < GL.width and y1 >= 0 and y1 < GL.height:
                        gpu.GPU.draw_pixel([int(x1), int(y1)], gpu.GPU.RGB8, color)
                    
                    # Incrementa o erro pelo coef ang
                    err += slope
                    
                    # Se o erro passa do tamanho de meio pixel, eu incremento o y e ajusto o erro
                    # O valor de 0.5 foi escolhido para deixar mais smooth e evitar desalinhamentos
                    # Foi testado comparar com 1, mas as retas resultantes não eram bem alinhadas,
                    # especialmente para o octogono

                    if err >= 0.5:
                        err -= 1
                        y1 += slope_y
                    
                    x1 += slope_x
            # Caso onde o y cresce mais
            else:
                # O coef eh "invertido" pois é como se rotacionasse a tela e seguisse o mesmo procedimento
                slope = dx/dy if dy != 0 else 1
                err = 0

                while int(y1) != int(y2):
                    if x1 >= 0 and x1 < GL.width and y1 >= 0 and y1 < GL.height:
                        gpu.GPU.draw_pixel([int(x1), int(y1)], gpu.GPU.RGB8, color)

                    err += slope

                    # Se o erro passa do tamanho de meio pixel, incremento o x e ajusto o erro
                    if err > 0.5:
                        err -= 1
                        x1 += slope_x
                    
                    y1 += slope_y

            # Pinta o ponto final
            if x2 >= 0 and x2 < GL.width and y2 >= 0 and y2 < GL.height:
                gpu.GPU.draw_pixel([int(x2), int(y2)], gpu.GPU.RGB8, color)


    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        color = [int(255 * colors['emissiveColor'][i]) for i in range(len(colors['emissiveColor']))]

        for x in range(0, round(radius) + 1):
            for y in range(0, round(radius) + 1):
                print(x, y)
                if radius**2 - 20 <= x**2 + y**2 <= radius**2 + 20 \
                    and x >= 0 and x < GL.width and y >= 0 and y < GL.height:
                    gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)


    @staticmethod
    def _drawTriangles(points, colors=None, colorPerVertex=False, vertexColors=None):
        # Configs
        width = GL.width
        height = GL.height
        sampling = GL.sampling
        width_sampling = width * sampling
        height_sampling = height * sampling

        if not colorPerVertex:
            color = np.array([int(255 * colors['emissiveColor'][i]) for i in range(len(colors['emissiveColor']))])
        else:
            color = None

        for i in range(0, len(points), 3):
            # Separa os vertices
            p1, p2, p3 = points[i:i+3]
            x1, y1, z1 = p1
            x2, y2, z2 = p2
            x3, y3, z3 = p3
            
            # Cria a otimizacao da caixa ao redor dos vertices 
            for x in range(int(min([x1, x2, x3])), int(max([x1, x2, x3])) + 1):
                for y in range(int(min([y1, y2, y3])), int(max([y1, y2, y3])) + 1):
                    # Por todos os pixels, apenas o desenha se estiver dentro da tela e se o centro do pixel obedecer
                    # a formula da reta normal
                    if GL._inside(
                        [x1, y1, x2, y2, x3, y3],
                        x + 0.5,
                        y + 0.5
                    ) and x >= 0 and x < width_sampling and y >= 0 and y < height_sampling:
                        # Interpolacao baricentrica
                        alpha, beta, gamma = GL._barycentric([x1, y1, x2, y2, x3, y3], [x + 0.5, y + 0.5])

                        # Interpolação para descobrir o Z do ponto
                        z = 1/(alpha/z1 + beta/z2 + gamma/z3)

                        if GL.z_buffer[y, x] > z:
                            GL.z_buffer[y, x] = z

                            transparency = float(colors.get('transparency', 1))

                            last_color = np.array(GL.sample_frame_buffer[y, x]) * transparency

                            if color is not None:
                                GL.sample_frame_buffer[y, x] = color * (1 - transparency) + last_color
                                #gpu.GPU.draw_pixel([int(x), int(y)], gpu.GPU.RGB8, color)
                            else:
                                rgb1, rgb2, rgb3 = vertexColors[i], vertexColors[i+1], vertexColors[i+2]
                                                            
                                r = (alpha * rgb1[0] / z1 + beta * rgb2[0] / z2 + gamma * rgb3[0] / z3) * z
                                g = (alpha * rgb1[1] / z1 + beta * rgb2[1] / z2 + gamma * rgb3[1] / z3) * z
                                b = (alpha * rgb1[2] / z1 + beta * rgb2[2] / z2 + gamma * rgb3[2] / z3) * z

                                pointColor = np.array([int(r * 255),
                                            int(g * 255),
                                            int(b * 255)])

                                GL.sample_frame_buffer[y, x] = pointColor * (1 - transparency) + last_color
                                #gpu.GPU.draw_pixel([int(x), int(y)], gpu.GPU.RGB8, pointColor)
            
        GL._drawPixels(width, height, sampling)
    

    @staticmethod
    def _drawPixels(width, height, sampling):
        # Mapear de volta o frame_buffer super sampled para o menor
        print(width, height)
        print(GL.sample_frame_buffer.shape)
        for x in range(width):
            for y in range(height):
                x_sampling = x * sampling
                y_sampling = y * sampling

                mean_color = np.mean(GL.sample_frame_buffer[
                                                y_sampling:y_sampling+sampling,
                                                x_sampling:x_sampling+sampling
                                            ], axis=(0, 1))
                
                gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, mean_color.astype(int))
    

    @staticmethod
    def _inside(vertices, x, y):
        """Função auxiliar para verificar se um ponto está "dentro de um lado" do triângulo."""
        
        # Formula da reta normal
        def L(x, y, x0, y0, x1, y1):
            return (x - x0)*(y1 - y0) - (y - y0)*(x1 - x0) >= 0

        # Se esta para todos os lados, podemos desenhar na tela
        return L(x, y, vertices[0], vertices[1], vertices[2], vertices[3]) and \
                L(x, y, vertices[2], vertices[3], vertices[4], vertices[5]) and \
                L(x, y, vertices[4], vertices[5], vertices[0], vertices[1])
    

    @staticmethod
    def _barycentric(vertices, point):
        """Função auxiliar para calcular as coordenadas baricentricas de um ponto em um triângulo."""
        x1, y1, x2, y2, x3, y3 = vertices
        x, y = point

        A1 = (x*(y2 - y3) + x2*(y3 - y) + x3*(y - y2)) / 2
        A2 = (x1*(y - y3) + x*(y3 - y1) + x3*(y1 - y)) / 2
        A3 = (x1*(y2 - y) + x2*(y - y1) + x*(y1 - y2)) / 2
        Atotal = A1 + A2 + A3

        alpha = A1 / Atotal
        beta = A2 / Atotal
        gamma = 1 - alpha - beta

        return alpha, beta, gamma
    

    @staticmethod
    def _drawTriangles3D(point, colors=None, colorPerVertex=False, vertexColors=None):
        vertices = []
        # Configs
        width = GL.width
        height = GL.height

        # High-resolution framebuffer dimensions
        high_res_width = width * GL.sampling
        high_res_height = height * GL.sampling

        # Para cada um dos triangulos
        for i in range(0, len(point), 9):
            # Separa os vertices
            x1, y1, z1, x2, y2, z2, x3, y3, z3 = point[i:i+9]

            triangle = np.array([
                [x1, x2, x3],
                [y1, y2, y3],
                [z1, z2, z3],
                [1, 1, 1]
            ])

            triangle = GL.perspective_matrix @ GL.viewpoint_matrix @ GL.transformation_stack[-1] @ triangle

            # Extract z-values before applying the mapping matrix (i.e., in original 3D space)
            z_values = triangle[2, :]

            # Normalizando a coordenada homogenea
            triangle = triangle / triangle[3, :]

            # Mapping matrix to move to screen space
            mapping_matrix = np.array([
                [high_res_width / 2, 0, 0, high_res_width / 2],
                [0, -high_res_height / 2, 0, high_res_height / 2],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

            # Apply the mapping to screen space for the x and y coordinates
            final_triangle = mapping_matrix @ triangle

            # Append the (x, y, z) tuples with z
            for j in range(3):
                vertices.append((final_triangle[0, j], final_triangle[1, j], z_values[j]))


        GL._drawTriangles(vertices, colors, colorPerVertex, vertexColors)


    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#TriangleSet2D
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        points = []
        for i in range(0, len(vertices), 2):
            x, y = vertices[i], vertices[i+1]
            # Map the 2D coordinates to screen space
            screen_x = x * GL.sampling
            screen_y = y * GL.sampling
            points.append([screen_x, screen_y, 0])

        GL._drawTriangles(points, colors)
            

    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.

        GL._drawTriangles3D(point, colors)

        

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        # Configs
        width = GL.width * GL.sampling
        height = GL.height * GL.sampling

        # Calculating the camera matrices
        ax, ay, az, theta = orientation
        R = np.linalg.inv(GL._quaternion_rotation(theta, [ax, ay, az]))
        T = np.array([
            [1, 0, 0, -position[0]],
            [0, 1, 0, -position[1]],
            [0, 0, 1, -position[2]],
            [0, 0, 0, 1]
        ])

        GL.viewpoint_matrix = R @ T

        # Calculating the perspective matrix
        fov_y = 2 * np.arctan(np.tan(fieldOfView/2) * height/np.sqrt(height**2 + width**2))

        aspect = width / height
        top = GL.near * np.tan(fov_y)
        right = top * aspect

        P = np.array([
            [GL.near/right, 0, 0, 0],
            [0, GL.near/top, 0, 0],
            [0, 0, -(GL.far+GL.near)/(GL.far-GL.near), -2*GL.far*GL.near/(GL.far-GL.near)],
            [0, 0, -1, 0]
        ])

        GL.perspective_matrix = P
    
    @staticmethod
    def _quaternion_rotation(theta, u):
        qr = np.cos(theta/2)
        qx = np.sin(theta/2) * u[0]
        qy = np.sin(theta/2) * u[1]
        qz = np.sin(theta/2) * u[2]

        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2 * (qx*qy - qz*qr), 2*(qx*qz + qy*qr), 0],
            [2*(qx*qy + qz*qr), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qr), 0],
            [2*(qx*qz - qy*qr), 2 * (qy*qz + qx*qr), 1 - 2*(qx**2 + qy**2), 0],
            [0, 0, 0, 1]
        ])

        return R

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas. 
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.

        T, S, R = np.identity(4), np.identity(4), np.identity(4)        

        if translation:
            T = np.array([
                [1, 0, 0, translation[0]],
                [0, 1, 0, translation[1]],
                [0, 0, 1, translation[2]],
                [0, 0, 0, 1]
            ])
        if scale:
            S = np.array([
                [scale[0], 0, 0, 0],
                [0, scale[1], 0, 0],
                [0, 0, scale[2], 0],
                [0, 0, 0, 1]
            ])
        if rotation:
            R = GL._quaternion_rotation(rotation[3], [rotation[0], rotation[1], rotation[2]])

        transformation_matrix = T @ R @ S

        GL.transformation_stack.append(GL.transformation_stack[-1] @ transformation_matrix)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        if GL.transformation_stack:
            GL.transformation_stack.pop()

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.
        
        idx = 0
        vertices = []
        for i, strip in enumerate(stripCount):
            strip_points = point[idx:idx + strip*3]
            idx += strip * 3

            # Process every triangle in the strip
            for j in range(0, len(strip_points) - 6, 3):
                # For odd-indexed triangles, reverse the last two vertices
                if (j // 3) % 2 == 1:
                    vertices.extend([
                        strip_points[j], strip_points[j + 1], strip_points[j + 2],
                        strip_points[j + 6], strip_points[j + 7], strip_points[j + 8],  
                        strip_points[j + 3], strip_points[j + 4], strip_points[j + 5] 
                    ])
                else:
                    vertices.extend(strip_points[j:j + 9])

        GL._drawTriangles3D(vertices, colors)


    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        vertices = []
        for i in range(len(index)):
            # End of strip
            if -1 in index[i:i+3]:
                continue
                
            v0, v1, v2 = index[i], index[i + 1], index[i + 2]
            
            # Switching orientation for every 2 triangles
            if i % 2 == 1:
                v0, v1, v2 = v1, v0, v2

            # Extract vertex coordinates
            triangle = []
            triangle.extend(point[v0*3:v0*3+3])
            triangle.extend(point[v1*3:v1*3+3])
            triangle.extend(point[v2*3:v2*3+3])

            vertices.extend(triangle)
        
        GL._drawTriangles3D(vertices, colors)  
        

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        vertices = []
        vertexColors = []
        v0 = coordIndex[0]

        # Tem algum bug que alguns que não deveriam ser colorPerVertex estão mandando True mas sem lista de colors
        if color is None: colorPerVertex = False

        c0 = None
        if colorPerVertex:
            c0 = colorIndex[0]

        i = 1
        while i < len(coordIndex):
            # Add the coordintes of the three points
            vertices.extend(coord[v0*3:v0*3+3])
            vertices.extend(coord[coordIndex[i]*3:coordIndex[i]*3+3])
            vertices.extend(coord[coordIndex[i+1]*3:coordIndex[i+1]*3+3])

            if colorPerVertex:
                vertexColors.append(color[c0*3:c0*3+3])
                vertexColors.append(color[colorIndex[i]*3:colorIndex[i]*3+3])
                vertexColors.append(color[colorIndex[i+1]*3:colorIndex[i+1]*3+3])

            if coordIndex[i+2] == -1:
                i += 3
                if i >= len(coordIndex):
                    break
                v0 = coordIndex[i]
                if colorPerVertex:
                    c0 = colorIndex[i]
            
            i += 1

        GL._drawTriangles3D(vertices, colors, colorPerVertex, vertexColors)


    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
