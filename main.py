import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import math
import random

import sys
sys.setrecursionlimit(50000)


def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    blur_img = cv2.blur(img, (10, 10))
    # cv2.imshow("", blur_img)
    # cv2.waitKey(0)
    return blur_img



def make_grid(img, cell_size = 7):
    h, w = img.shape
    dark_grid = np.zeros((h // cell_size, w // cell_size), dtype=np.float32)
    
    for i in range(0, h, cell_size):
        for j in range (0, w, cell_size):
            cell = img[i:i+cell_size, j:j+cell_size]
            
            if cell.shape[0] < cell_size or cell.shape[1] < cell_size:
                continue
            
            luminosity = np.mean(cell)
            darkness = 255-luminosity
            dark_grid[i // cell_size, j // cell_size] = darkness
            
    min_val = np.min(dark_grid)
    maxwell = np.max(dark_grid) # Glenn 
    levels = 4
    
    def min_max_norm(x):
        return math.floor((x - min_val)*(levels-1) / (maxwell - min_val))
    
    def prob_val(x):
        norm_val = min_max_norm(x)
        if norm_val == 0:
            return 0 
        elif norm_val == 1:
            return 1 if random.random() < 0.5 else 0
        elif norm_val == 2:
            return 3
        elif norm_val == 3:
            return 6
        else:
            return norm_val
    
    vectorized_min_max = np.vectorize(prob_val)
    
    norm_dark_grid = vectorized_min_max(dark_grid)
    # plt.imshow(levels-norm_dark_grid, cmap='gray')
    # plt.show()
    # print(norm_dark_grid)
    return norm_dark_grid
    



def dfs(grid, x, y, visited, path):
    if (x<0 or x >= grid.shape[0] or y<0 or y >= grid.shape[1] or visited[x, y] >= grid[x, y]):
        return

    visited[x, y] += 1
    path.append((x, y))
    
    moves = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (-1, -1)]
    random.shuffle(moves)
    
    for move in moves:
        new_x, new_y = x + move[0], y + move[1]
        dfs(grid, new_x, new_y, visited, path)
        


def get_path(grid):
    n, m = grid.shape
    visited = np.zeros(grid.shape, dtype=int)
    path = []
    
    for i in range(n):
        for j in range(m):
            while visited[i, j] < grid[i, j]:
                dfs(grid, i, j, visited, path)
    
    return path



def perturb_path(path, perturbation=0.25):
    result_path = []
    
    for x,y in path:
        x_perturb = np.random.uniform(-perturbation, perturbation)
        y_perturb = np.random.uniform(-perturbation, perturbation)

        new_x = x + x_perturb
        new_y = y + y_perturb

        result_path.append((new_x, new_y))
        
    return result_path


def draw_spline(path):
    y = [i[0] for i in path]
    x = [i[1] for i in path]
    
    x = np.array(x)
    y = np.array(y)
    
    # plt.plot(x, y)
    
    data = np.array([x, y])
    
    tck, u = interpolate.splprep(data, s=0)
    
    unew = np.arange(0, 1, 0.00001)
    out = interpolate.splev(unew, tck)
        
    plt.plot(out[0], out[1], color='black', label='Spline', linewidth=0.5)
    # plt.legend()
    plt.axis('equal')
    plt.savefig('plot.png', dpi=700)
    plt.show()
    
    



def main(path) :
    processedimg = preprocess(path)
    norm_dark_grid = make_grid(processedimg)
    path = get_path(norm_dark_grid)
    
    perturbed_path = perturb_path(path)

    draw_spline(perturbed_path)
    
    
main('akash.jpg')